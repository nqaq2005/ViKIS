import yaml
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.embeddings.visual_encoder import VisualEncoder
from src.embeddings.text_encoder import TextEncoder
from src.retrieval.reranker import TextReranker
from src.utils.file_utils import getenv

class VideoKISRetriever:
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        qdrant_config_path: str = "configs/qdrant_config.yaml",
        models_config_path: str = "configs/models_config.yaml"
    ):
        # 1. Tải cấu hình
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        with open(qdrant_config_path, "r", encoding="utf-8") as f:
            self.qdrant_cfg = yaml.safe_load(f)

        self.retrieval_cfg = self.config.get("retrieval", {})

        collections_cfg = self.qdrant_cfg.get("collections", {})
        self.visual_col = collections_cfg.get("visual_keyframes", {}).get("name", "visual_keyframes")
        self.transcript_col = collections_cfg.get("transcript_segments", {}).get("name", "transcript_segments")

        # 2. Khởi tạo Qdrant Client
        self.qdrant_url = getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = getenv("QDRANT_API_KEY", None)
        
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

        # 3. Khởi tạo Models (Dùng chung bộ models đã thiết lập)
        print("[RETRIEVER] Đang nạp Visual Encoder (Jina-CLIP) cho Text-to-Image Search...")
        self.visual_encoder = VisualEncoder()
        
        print("[RETRIEVER] Đang nạp Text Encoder (BGE-M3) cho Hybrid Text Search...")
        self.text_encoder = TextEncoder()

        print("[RETRIEVER] Đang nạp Cross-Encoder (Reranker)...")
        self.reranker = TextReranker(config_path=models_config_path)

    def search_visual(self, query_text: str, retrieve_top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Tìm kiếm khung hình video bằng câu truy vấn văn bản (Text-to-Image).
        """
        k = retrieve_top_k 
        # Mã hóa câu hỏi tiếng Việt sang không gian hình ảnh 1024d
        query_vector = self.visual_encoder.encode_text_query(query_text)

        # Truy vấn Qdrant (Query API mới nhất)
        response = self.client.query_points(
            collection_name=self.visual_col,
            query=query_vector,          # đổi từ query_vector -> query
            limit=k,
            with_payload=True
        )

        points = getattr(response, "points", None) or []

        formatted_results = []
        for res in points:
            payload = getattr(res, "payload", None) or {}
            formatted_results.append({
                "video_id": payload.get("video_id"),
                "scene_id": payload.get("scene_id"),
                "timestamp": payload.get("timestamp"),
                "frame_path": payload.get("frame_path"),
                "ocr_text": payload.get("ocr_text", ""),
                "score": getattr(res, "score", 0.0),
                "source": "visual"
            })
        return formatted_results
    
    def search_transcript(self, query_text: str, retrieve_top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Tìm kiếm lời thoại video bằng Hybrid Search (Dense + Sparse SPLADE).
        """
        k = retrieve_top_k 
        
        # Mã hóa câu hỏi sang Dense (Semantic) và Sparse (Lexical/Keyword)
        query_vectors = self.text_encoder.encode_query(query_text)
        
        # Qdrant Prefetch cho Hybrid Search
        prefetch_queries = [
            models.Prefetch(
                query=query_vectors["dense"],
                using="dense",
                limit=k
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=query_vectors["sparse"]["indices"],
                    values=query_vectors["sparse"]["values"]
                ),
                using="sparse",
                limit=k
            )
        ]

        # Thực thi tìm kiếm gộp (Reciprocal Rank Fusion (RRF) nội bộ của Qdrant)
        response = self.client.query_points(
            collection_name=self.transcript_col,
            prefetch=prefetch_queries,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True
        )

        results = getattr(response, "points", None) or []

        formatted_results = []
        for res in results:
            payload = getattr(res, "payload", None) or {}
            formatted_results.append({
                "video_id": payload.get("video_id"),
                "start_time": payload.get("start_time"),
                "end_time": payload.get("end_time"),
                "text": payload.get("text", ""),
                "score": getattr(res, "score", 0.0),
                "source": "transcript"
            })

        if formatted_results:
            formatted_results = self.reranker.rerank_transcripts(
                query=query_text, 
                hits=formatted_results, 
                top_k=k
            )
            
        return formatted_results

    def retrieve(self, query_text: str, retrieve_top_k: int = 100) -> Dict[str, List[Dict[str, Any]]]:

        print(f"\n[RETRIEVER] Đang xử lý truy vấn: '{query_text}' với độ sâu {retrieve_top_k}")
        
        visual_hits = self.search_visual(query_text, retrieve_top_k=retrieve_top_k)
        transcript_hits = self.search_transcript(query_text, retrieve_top_k=retrieve_top_k)
        
        return {
            "visual": visual_hits,
            "transcript": transcript_hits
        }


if __name__ == "__main__":
    retriever = VideoKISRetriever()
    test_q = "kế hoạch doanh thu quý 3"
    results = retriever.retrieve(test_q)
    
    print("\n--- TOP 3 VISUAL HITS ---")
    for r in results["visual"][:3]:
        print(f"[{r['video_id']} - {r['timestamp']}s] Score: {r['score']:.4f}")
        
    print("\n--- TOP 3 TRANSCRIPT HITS ---")
    for r in results["transcript"][:3]:
        print(f"[{r['video_id']} - {r['start_time']}s] Score: {r['score']:.4f} | Text: {r['text'][:50]}...")