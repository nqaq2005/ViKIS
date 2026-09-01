import yaml
import torch
from typing import List, Dict, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TextReranker:
    def __init__(self, config_path: str = "configs/models_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Cấu hình Reranker (nếu chưa có trong yaml, dùng mặc định)
        reranker_cfg = self.config.get("reranker", {})
        self.model_name = reranker_cfg.get("model_id", "BAAI/bge-reranker-v2-m3")
        self.max_length = reranker_cfg.get("max_length", 512)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[RERANKER] Đang tải mô hình Cross-Encoder: '{self.model_name}' trên {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        print("[RERANKER] Tải mô hình thành công!")

    def rerank_transcripts(self, query: str, hits: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Chấm điểm lại (Rerank) danh sách các transcript/lời thoại.
        Cross-Encoder sẽ chú ý (attention) chéo giữa từng từ trong Query và Text.
        """
        if not hits:
            return []

        # Chuẩn bị dữ liệu đầu vào dạng cặp (Query, Document)
        pairs = []
        for hit in hits:
            text = hit.get("text", "")
            pairs.append([query, text])

        # Inference không lưu đồ thị đạo hàm để tiết kiệm RAM/VRAM
        with torch.inference_mode():
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Mô hình Cross-Encoder trả về 1 logit score duy nhất cho mỗi cặp
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().tolist()

        # Cập nhật điểm mới vào danh sách hits
        for i, hit in enumerate(hits):
            # Lưu lại điểm gốc từ Qdrant để đối chiếu nếu cần
            hit["qdrant_score"] = hit.get("score", 0.0)
            
            # Ghi đè score bằng điểm Rerank (điểm này chuẩn xác hơn điểm cosine của Qdrant)
            hit["score"] = scores[i]

        # Sắp xếp lại theo điểm Rerank giảm dần
        reranked_hits = sorted(hits, key=lambda x: x["score"], reverse=True)
        
        # Cắt lấy Top K nếu được yêu cầu
        if top_k:
            reranked_hits = reranked_hits[:top_k]
            
        return reranked_hits

if __name__ == "__main__":
    # Test mô phỏng
    reranker = TextReranker()
    q = "kế hoạch doanh thu quý 3"
    
    mock_hits = [
        {"video_id": "vid_01", "text": "hôm nay chúng ta sẽ bàn về thời tiết", "score": 0.5},
        {"video_id": "vid_02", "text": "báo cáo tài chính cho thấy doanh thu tăng trưởng", "score": 0.6},
        {"video_id": "vid_03", "text": "mục tiêu doanh số trong quý 3 là 500 tỷ", "score": 0.4} # Qdrant có thể nhầm và cho điểm thấp
    ]
    
    print("\n--- TRƯỚC KHI RERANK ---")
    for r in mock_hits:
        print(f"[{r['video_id']}] Score: {r['score']} | {r['text']}")
        
    reranked = reranker.rerank_transcripts(q, mock_hits)
    
    print("\n--- SAU KHI RERANK (Cross-Encoder) ---")
    for r in reranked:
        print(f"[{r['video_id']}] Rerank Score: {r['score']:.4f} | {r['text']}")