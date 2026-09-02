import yaml
import torch
import numpy as np
from typing import List, Dict, Any, Mapping
from FlagEmbedding import BGEM3FlagModel
from src.utils.fix_bug_FlagEmbedding import fix_bug_FlagEmbedding

fix_bug_FlagEmbedding() # bug của thư viện 
  
class TextEncoder:
    def __init__(self, models_config_path: str = "configs/models_config.yaml"):
        with open(models_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.cfg = config.get("text_model", {})
        self.model_id = self.cfg.get("model_id", "BAAI/bge-m3")
        self.device = self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.cfg.get("batch_size", 16)
        self.max_length = self.cfg.get("max_length", 8192)
        self.normalize = self.cfg.get("normalize", True)

        print(f"[TEXT ENCODER] Đang tải mô hình Hybrid {self.model_id} trên {self.device}...")
        self.model = BGEM3FlagModel(
            self.model_id,
            use_fp16="cuda" in self.device,
            device=self.device
        )

    def _as_dense_vector(self, value: Any) -> np.ndarray:
        """Ép dense vector thành numpy.ndarray trước khi normalize."""
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _as_lexical_weights(self, value: Any) -> Dict[str, float]:
        """Ép sparse vector thành dict[str, float]."""
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return {str(k): float(v) for k, v in value.items()}
        return {str(k): float(v) for k, v in dict(value).items()}

    def _format_sparse_for_qdrant(self, lexical_weights: Dict[str, float]) -> Dict[str, List]:
        """
        Chuyển đổi lexical weights dạng token_id -> weight sang format SparseVector của Qdrant:
        {"indices": [int, ...], "values": [float, ...]}
        """
        indices = []
        values = []

        for token_id, weight in lexical_weights.items():
            indices.append(int(token_id))
            values.append(float(weight))

        return {
            "indices": indices,
            "values": values
        }

    def encode_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Mã hóa danh sách các đoạn transcript/văn bản thành cả Dense và Sparse vectors.
        
        :param texts: Danh sách câu thoại / văn bản
        :return: Danh sách dict [{'dense': [...], 'sparse': {'indices': [...], 'values': [...]}}]
        """
        if not texts:
            return []

        outputs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )

        dense_vecs = outputs["dense_vecs"]
        sparse_vecs = outputs["lexical_weights"]

        results = []
        for i in range(len(texts)):
            d_vec = self._as_dense_vector(dense_vecs[i])
            if self.normalize:
                norm = np.linalg.norm(d_vec)
                if norm > 0:
                    d_vec = d_vec / norm

            s_vec = self._format_sparse_for_qdrant(self._as_lexical_weights(sparse_vecs[i]))

            results.append({
                "dense": d_vec.tolist(),
                "sparse": s_vec
            })

        return results

    def encode_query(self, query: str) -> Dict[str, Any]:
        """
        Mã hóa một câu query phục vụ Hybrid Search (Dense + Sparse) trên transcript_segments.
        """
        outputs = self.model.encode(
            [query],
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )

        d_vec = self._as_dense_vector(outputs["dense_vecs"][0])
        if self.normalize:
            norm = np.linalg.norm(d_vec)
            if norm > 0:
                d_vec = d_vec / norm

        s_vec = self._format_sparse_for_qdrant(self._as_lexical_weights(outputs["lexical_weights"][0]))

        return {
            "dense": d_vec.tolist(),
            "sparse": s_vec
        }


if __name__ == "__main__":
    encoder = TextEncoder()
    test_text = "kế hoạch tăng trưởng doanh thu quý 3 năm 2026 đạt 25%"
    
    # Test encode doc
    doc_res = encoder.encode_documents([test_text])[0]
    print(f"Kích thước Dense Vector: {len(doc_res['dense'])}")
    print(f"Số lượng tokens trong Sparse Vector: {len(doc_res['sparse']['indices'])}")
    print(f"Mẫu Sparse Vector: indices={doc_res['sparse']['indices'][:5]}, values={doc_res['sparse']['values'][:5]}")