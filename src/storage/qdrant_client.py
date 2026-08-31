import os
import uuid
import yaml
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

class VideoKISQdrantClient:
    def __init__(self, config_path: str = "configs/qdrant_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        conn_cfg = self.config.get("connection", {})
        self.url = os.getenv("QDRANT_URL", conn_cfg.get("url", "http://localhost:6333"))
        self.api_key = os.getenv("QDRANT_API_KEY", conn_cfg.get("api_key", None))
        self.timeout = conn_cfg.get("timeout", 30)

        # Khởi tạo kết nối Qdrant
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout
        )

        collections_cfg = self.config.get("collections", {})
        self.visual_col = collections_cfg.get("visual_keyframes", {}).get("name", "visual_keyframes")
        self.transcript_col = collections_cfg.get("transcript_segments", {}).get("name", "transcript_segments")

    def _generate_point_id(self, key_str: str) -> str:
        """Tạo UUID xác định (deterministic UUID5) từ chuỗi định danh để tránh trùng lặp dữ liệu."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key_str))

    # =========================================================================
    # 1. BATCH UPSERT: visual_keyframes
    # =========================================================================
    def upsert_keyframes(
        self,
        keyframes: List[Dict[str, Any]],
        vectors: List[List[float]],
        batch_size: int = 64
    ) -> int:
        """
        Nạp danh sách keyframe và visual vector (Jina-CLIP-v2) vào collection visual_keyframes.
        
        :param keyframes: List dict chứa video_id, scene_id, timestamp, frame_path, ocr_text, sharpness
        :param vectors: Danh sách vector dense 1024d tương ứng
        :param batch_size: Kích thước batch đẩy lên Qdrant
        :return: Tổng số point đã upsert thành công
        """
        if not keyframes or not vectors or len(keyframes) != len(vectors):
            raise ValueError("Độ dài của keyframes và vectors phải bằng nhau và không được rỗng.")

        points = []
        for kf, vec in zip(keyframes, vectors):
            # Tạo unique deterministic ID dựa trên video_id + timestamp
            point_id = self._generate_point_id(f"{kf['video_id']}_scene_{kf.get('scene_id', 0)}_{kf['timestamp']}")
            
            payload = {
                "video_id": str(kf["video_id"]),
                "scene_id": int(kf.get("scene_id", 0)),
                "timestamp": float(kf["timestamp"]),
                "frame_path": str(kf.get("frame_path", "")),
                "ocr_text": str(kf.get("ocr_text", "")),
                "sharpness": float(kf.get("sharpness", 0.0))
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload
                )
            )

        # Đẩy theo từng batch
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.visual_col,
                points=batch,
                wait=True
            )
            total_upserted += len(batch)

        print(f"[QDRANT] Đã upsert {total_upserted} points vào collection '{self.visual_col}'.")
        return total_upserted

    # =========================================================================
    # 2. BATCH UPSERT: transcript_segments
    # =========================================================================
    def upsert_transcripts(
        self,
        video_id: str,
        transcripts: List[Dict[str, Any]],
        embeddings: List[Dict[str, Any]],
        batch_size: int = 64
    ) -> int:
        """
        Nạp danh sách transcript và hybrid vectors (BGE-M3 Dense + Sparse) vào collection transcript_segments.
        
        :param video_id: ID của video
        :param transcripts: List dict chứa start, end, text
        :param embeddings: List dict chứa {'dense': [...], 'sparse': {'indices': [...], 'values': [...]}}
        :param batch_size: Kích thước batch
        :return: Tổng số point đã upsert thành công
        """
        if not transcripts or not embeddings or len(transcripts) != len(embeddings):
            raise ValueError("Độ dài transcripts và embeddings phải bằng nhau và không được rỗng.")

        points = []
        for tr, emb in zip(transcripts, embeddings):
            start_t = float(tr["start"])
            end_t = float(tr["end"])
            point_id = self._generate_point_id(f"{video_id}_trans_{start_t}_{end_t}")

            payload = {
                "video_id": str(video_id),
                "start_time": start_t,
                "end_time": end_t,
                "text": str(tr.get("text", ""))
            }

            # Named vectors: Dense vector + Sparse vector của BGE-M3
            named_vectors = {
                "dense": emb["dense"],
                "sparse": models.SparseVector(
                    indices=emb["sparse"]["indices"],
                    values=emb["sparse"]["values"]
                )
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=named_vectors,
                    payload=payload
                )
            )

        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.transcript_col,
                points=batch,
                wait=True
            )
            total_upserted += len(batch)

        print(f"[QDRANT] Đã upsert {total_upserted} points vào collection '{self.transcript_col}'.")
        return total_upserted

    # =========================================================================
    # 3. QUẢN LÝ DỮ LIỆU (CLEANUP / DELETE)
    # =========================================================================
    def delete_by_video_id(self, video_id: str):
        """Xóa toàn bộ dữ liệu visual và transcript của một video (hỗ trợ re-indexing)."""
        video_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id)
                )
            ]
        )

        for col in [self.visual_col, self.transcript_col]:
            if self.client.collection_exists(col):
                self.client.delete(
                    collection_name=col,
                    points_selector=models.FilterSelector(filter=video_filter),
                    wait=True
                )
                print(f"[QDRANT] Đã xóa dữ liệu video '{video_id}' khỏi '{col}'.")


if __name__ == "__main__":
    q_client = VideoKISQdrantClient()
    
    # Test mẫu upsert visual keyframes
    mock_keyframes = [{
        "video_id": "test_vid_01",
        "scene_id": 1,
        "timestamp": 10.5,
        "frame_path": "data/keyframes/test_vid_01/shot_0001_kf01.jpg",
        "ocr_text": "DOANH THU QUY 3",
        "sharpness": 150.2
    }]
    mock_visual_vecs = [[0.01] * 1024]
    q_client.upsert_keyframes(mock_keyframes, mock_visual_vecs)

    # Test mẫu upsert transcript segments
    mock_transcripts = [{
        "start": 10.0,
        "end": 15.2,
        "text": "báo cáo tài chính quý ba ghi nhận tăng trưởng vượt bậc"
    }]
    mock_text_embeddings = [{
        "dense": [0.02] * 1024,
        "sparse": {"indices": [101, 204], "values": [0.8, 0.4]}
    }]
    q_client.upsert_transcripts("test_vid_01", mock_transcripts, mock_text_embeddings)