import os
import json
import yaml
from typing import List, Dict, Any
import easyocr

class OCRScanner:
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        models_config_path: str = "configs/models_config.yaml",
        output_dir: str = ""
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        with open(models_config_path, "r", encoding="utf-8") as f:
            self.models_config = yaml.safe_load(f)

        
        # Cấu hình OCR
        self.ocr_sys_cfg = self.config.get("ingestion", {}).get("ocr", {})
        self.ocr_model_cfg = self.models_config.get("ocr_model", {})
        
        self.enabled = self.ocr_sys_cfg.get("enabled", True)
        self.min_confidence = self.ocr_sys_cfg.get("min_confidence", 0.5)
        if output_dir != "":
            self.cache_dir = output_dir
        else:
            self.cache_dir = self.config.get("paths", {}).get("ocr_cache_dir", "data/ocr_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        if self.enabled:
            languages = self.ocr_model_cfg.get("languages", ["vi", "en"])
            use_gpu = self.ocr_model_cfg.get("use_gpu", True)
            print(f"[OCR] Khởi tạo EasyOCR với ngôn ngữ: {languages}, GPU={use_gpu}...")
            self.reader = easyocr.Reader(languages, gpu=use_gpu)
        else:
            self.reader = None

    def scan_image(self, image_path: str) -> str:
        """
        Quét văn bản trên 1 file ảnh keyframe.
        
        :param image_path: Đường dẫn tới file ảnh (.jpg)
        :return: Chuỗi văn bản tiếng Việt trích xuất được (đã lọc confidence)
        """
        if not self.enabled or self.reader is None or not os.path.exists(image_path):
            return ""

        try:
            # EasyOCR trả về dạng: [ (bbox, text, prob), ... ]
            results = self.reader.readtext(image_path)
            valid_words = [
                text.strip()
                for (_, text, prob) in results
                if prob >= self.min_confidence and len(text.strip()) > 1
            ]
            return " ".join(valid_words)
        except Exception as e:
            print(f"[OCR ERROR] Lỗi khi quét {image_path}: {e}")
            return ""

    def process_keyframes(
        self,
        video_id: str,
        keyframes: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Bổ sung trường ocr_text vào danh sách keyframe records của một video.
        """
        cache_path = os.path.join(self.cache_dir, f"{video_id}_ocr.json")

        # Đọc cache nếu có sẵn
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            # Map kết quả cache vào keyframes list
            cache_map = {item["frame_path"]: item.get("ocr_text", "") for item in cached_data}
            for kf in keyframes:
                kf["ocr_text"] = cache_map.get(kf["frame_path"], "")
            return keyframes

        # Quét mới
        for kf in keyframes:
            img_path = kf["frame_path"]
            kf["ocr_text"] = self.scan_image(img_path)

        # Lưu cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(keyframes, f, ensure_ascii=False, indent=2)

        print(f"[OCR] Đã quét và lưu cache OCR cho video '{video_id}' ({len(keyframes)} frames)")
        return keyframes
