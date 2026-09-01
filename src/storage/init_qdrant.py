import os
import yaml
from qdrant_client import QdrantClient
from qdrant_client.http import models

class QdrantInitializer:
    def __init__(self, config_path: str = "configs/qdrant_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = os.getenv("QDRANT_API_KEY", None)
        self.timeout = int(os.getenv("QDRANT_TIMEOUT", 60))

        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout
        )
        self.collections_cfg = self.config.get("collections", {})
        self.hnsw_cfg = self.config.get("hnsw_config", {})

    def _create_payload_indexes(self, col_name: str, index_list: list):
        """Tạo các trường index payload theo cấu hình."""
        for idx in index_list:
            field = idx["field_name"]
            schema_type = idx["schema_type"]
            
            if schema_type == "keyword":
                schema = models.PayloadSchemaType.KEYWORD
            elif schema_type == "integer":
                schema = models.PayloadSchemaType.INTEGER
            elif schema_type == "float":
                schema = models.PayloadSchemaType.FLOAT
            elif schema_type == "text":
                schema = models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.MULTILINGUAL,
                    lowercase=True
                )
            else:
                continue

            self.client.create_payload_index(
                collection_name=col_name,
                field_name=field,
                field_schema=schema
            )
            print(f"  └── Đã đánh index payload: {field} ({schema_type})")

    def initialize_collections(self, recreate: bool = False):
        """Khởi tạo toàn bộ collections: visual_keyframes và transcript_segments."""
        hnsw_diff = models.HnswConfigDiff(
            m=self.hnsw_cfg.get("m", 16),
            ef_construct=self.hnsw_cfg.get("ef_construct", 100),
            full_scan_threshold=self.hnsw_cfg.get("full_scan_threshold", 1000),
            on_disk=self.hnsw_cfg.get("on_disk", False)
        )

        # -----------------------------------------------------------------
        # 1. Khởi tạo visual_keyframes
        # -----------------------------------------------------------------
        vk_cfg = self.collections_cfg.get("visual_keyframes", {})
        vk_name = vk_cfg.get("name", "visual_keyframes")

        if recreate and self.client.collection_exists(vk_name):
            self.client.delete_collection(vk_name)
            print(f"[RESET] Đã xóa collection cũ: {vk_name}")

        if not self.client.collection_exists(vk_name):
            self.client.create_collection(
                collection_name=vk_name,
                vectors_config=models.VectorParams(
                    size=vk_cfg.get("vectors", {}).get("size", 1024),
                    distance=models.Distance.COSINE,
                    on_disk=vk_cfg.get("vectors", {}).get("on_disk", False)
                ),
                hnsw_config=hnsw_diff
            )
            print(f"[CREATED] Collection: {vk_name}")
            self._create_payload_indexes(vk_name, vk_cfg.get("payload_indexes", []))

        # -----------------------------------------------------------------
        # 2. Khởi tạo transcript_segments
        # -----------------------------------------------------------------
        ts_cfg = self.collections_cfg.get("transcript_segments", {})
        ts_name = ts_cfg.get("name", "transcript_segments")

        if recreate and self.client.collection_exists(ts_name):
            self.client.delete_collection(ts_name)
            print(f"[RESET] Đã xóa collection cũ: {ts_name}")

        if not self.client.collection_exists(ts_name):
            self.client.create_collection(
                collection_name=ts_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=ts_cfg.get("vectors", {}).get("dense", {}).get("size", 1024),
                        distance=models.Distance.COSINE,
                        on_disk=ts_cfg.get("vectors", {}).get("dense", {}).get("on_disk", False)
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=ts_cfg.get("sparse_vectors", {}).get("sparse", {}).get("on_disk", False)
                        )
                    )
                },
                hnsw_config=hnsw_diff
            )
            print(f"[CREATED] Collection: {ts_name}")
            self._create_payload_indexes(ts_name, ts_cfg.get("payload_indexes", []))

if __name__ == "__main__":
    init_tool = QdrantInitializer()
    init_tool.initialize_collections(recreate=True)