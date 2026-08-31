import os
import yaml
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Any, Optional
from transformers import AutoModel

class VisualEncoder:
    def __init__(self, models_config_path: str = "configs/models_config.yaml"):
        with open(models_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.cfg = config.get("visual_model", {})
        self.model_id = self.cfg.get("model_id", "jinaai/jina-clip-v2")
        self.device = self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dimension = self.cfg.get("dimension", 1024)
        self.batch_size = self.cfg.get("batch_size", 32)
        self.normalize = self.cfg.get("normalize", True)

        print(f"[VISUAL ENCODER] Đang tải mô hình {self.model_id} trên {self.device}...")
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
        ).to(self.device)
        self.model.eval()

    def encode_images(self, image_paths: List[str]) -> List[List[float]]:
        """
        Mã hóa danh sách đường dẫn ảnh thành danh sách dense vectors.
        """
        if not image_paths:
            return []

        all_embeddings = []

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            pil_images = []

            for path in batch_paths:
                if os.path.exists(path):
                    try:
                        img = Image.open(path).convert("RGB")
                        pil_images.append(img)
                    except Exception as e:
                        print(f"[VISUAL ENCODER ERROR] Không thể đọc ảnh {path}: {e}")
                        pil_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                else:
                    # Fallback ảnh đen nếu file bị thiếu
                    pil_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))

            with torch.no_grad():
                # jina-clip-v2 hỗ trợ encode_image với truncate_dim (Matryoshka)
                vectors = self.model.encode_image(
                    pil_images,
                    truncate_dim=self.dimension
                )

                if self.normalize:
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    norms[norms == 0] = 1e-12
                    vectors = vectors / norms

                all_embeddings.extend(vectors.tolist())

        return all_embeddings

    def encode_text_query(self, query: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Mã hóa câu truy vấn text của người dùng để so khớp trong không gian Visual (Text-to-Visual search).
        """
        is_single = isinstance(query, str)
        queries = [query] if is_single else query

        with torch.no_grad():
            vectors = self.model.encode_text(
                queries,
                truncate_dim=self.dimension
            )

            if self.normalize:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1e-12
                vectors = vectors / norms

            result = vectors.tolist()

        return result[0] if is_single else result


if __name__ == "__main__":
    encoder = VisualEncoder()
    # Test encode query tiếng Việt
    test_query = "Người đàn ông mặc áo vest đang chỉ tay vào slide báo cáo tài chính"
    q_vec = encoder.encode_text_query(test_query)
    print(f"Kích thước vector Text-to-Visual: {len(q_vec)}")