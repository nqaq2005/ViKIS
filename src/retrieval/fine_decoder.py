import os
import yaml
import tempfile
import shutil
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from decord import VideoReader, cpu

from src.embeddings.visual_encoder import VisualEncoder

class FineDecoder:
    def __init__(
        self,
        visual_encoder: VisualEncoder,  # Dùng lại instance đã nạp trên RAM để tránh load model 2 lần
        config_path: str = "configs/config.yaml"
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.raw_videos_dir = self.config.get("paths", {}).get("raw_videos_dir", "data/raw_videos")
        
        coarse_cfg = self.config.get("retrieval", {}).get("coarse_to_fine", {})
        self.expansion_window = coarse_cfg.get("expansion_window_sec", 2.0)  # Mở rộng biên an toàn
        self.dense_fps = coarse_cfg.get("dense_fps", 5.0)  # Lấy 5 frames/giây để quét mịn
        
        self.visual_encoder = visual_encoder

    def _find_video_path(self, video_id: str) -> str:
        """Tìm đường dẫn file video gốc dựa trên video_id."""
        extensions = [".mp4", ".mkv", ".avi", ".webm", ".mov"]
        for ext in extensions:
            path = os.path.join(self.raw_videos_dir, f"{video_id}{ext}")
            if os.path.exists(path):
                return path
            # Thử chữ in hoa
            path_upper = os.path.join(self.raw_videos_dir, f"{video_id}{ext.upper()}")
            if os.path.exists(path_upper):
                return path_upper
        return ""

    def refine_search(
        self, 
        video_id: str, 
        start_sec: float, 
        end_sec: float, 
        query_text: str, 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Trích xuất dày đặc các khung hình trong khoảng thời gian [start, end] và chấm điểm lại.
        """
        video_path = self._find_video_path(video_id)
        if not video_path:
            print(f"[FINE DECODER ERROR] Không tìm thấy video gốc cho ID: {video_id}")
            return []

        # 1. Mở rộng vùng tìm kiếm để không bỏ lỡ hành động ở sát biên
        search_start = max(0.0, start_sec - self.expansion_window)
        search_end = end_sec + self.expansion_window

        # 2. Đọc video bằng Decord
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            video_duration = total_frames / fps
        except Exception as e:
            print(f"[FINE DECODER ERROR] Không thể đọc video {video_path}: {e}")
            return []

        # Đảm bảo search_end không vượt quá độ dài video
        search_end = min(search_end, video_duration)
        
        # Đảm bảo search_start không vượt quá search_end (chống dữ liệu đầu vào bị lệch/lớn hơn duration)
        search_start = min(search_start, search_end)

        # 3. Tính toán các mốc thời gian cần trích xuất (Dense Sampling)
        # Nếu dense_fps = 5, mỗi giây lấy 5 frames (cách nhau 0.2s)
        sample_interval = 1.0 / self.dense_fps
        sample_times = np.arange(search_start, search_end, sample_interval)
        
        if len(sample_times) == 0:
            sample_times = [(search_start + search_end) / 2.0]

        frame_indices = [min(int(t * fps), total_frames - 1) for t in sample_times]

        # 4. Trích xuất frame siêu tốc và lưu tạm
        temp_dir = tempfile.mkdtemp(prefix=f"kis_fine_{video_id}_")
        temp_image_paths = []
        
        print(f"[FINE DECODER] Đang quét mịn {len(frame_indices)} frames từ {search_start:.1f}s đến {search_end:.1f}s...")
        
        try:
            frames = vr.get_batch(frame_indices).asnumpy()
            
            for i, frame in enumerate(frames):
                img_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
                Image.fromarray(frame).save(img_path, quality=90)
                temp_image_paths.append(img_path)

            # 5. Mã hóa toàn bộ batch ảnh tạm bằng Visual Encoder (Jina-CLIP)
            img_vectors = self.visual_encoder.encode_images(temp_image_paths)
            img_vectors = np.array(img_vectors)  # Shape: (N, 1024)

            # 6. Mã hóa câu truy vấn
            text_vector = self.visual_encoder.encode_text_query(query_text)
            text_vector = np.array(text_vector)  # Shape: (1024,)

            # 7. Tính Cosine Similarity (Dot product vì vector đã normalize)
            # img_vectors: (N, 1024), text_vector: (1024,) -> dot product -> (N,)
            similarities = np.dot(img_vectors, text_vector)

            # 8. Sắp xếp và lấy Top K frame tốt nhất
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    "video_id": video_id,
                    "timestamp": round(float(sample_times[idx]), 3),
                    "score": float(similarities[idx]),
                    # Load ảnh dạng PIL hoặc Base64 để hiển thị UI, sau đó xóa file tạm
                    "frame_image": Image.open(temp_image_paths[idx]).copy() 
                })
                
        finally:
            # Luôn dọn dẹp thư mục tạm để tránh đầy ổ cứng
            shutil.rmtree(temp_dir, ignore_errors=True)

        return results


if __name__ == "__main__":
    # Test thử nghiệm độc lập
    print("Khởi tạo Visual Encoder...")
    encoder = VisualEncoder()
    decoder = FineDecoder(visual_encoder=encoder)
    
    test_video = "L22_V001"
    q = "người đàn ông mặc áo vest"
    
    print(f"Bắt đầu quét mịn Coarse-to-Fine cho video {test_video}...")
    # Giả sử Qdrant trả về khoảng khả nghi là từ giây 10.0 đến 12.0
    fine_results = decoder.refine_search(test_video, start_sec=10.0, end_sec=12.0, query_text=q, top_k=2)
    
    print("\n--- KẾT QUẢ QUÉT MỊN ---")
    for r in fine_results:
        print(f"Timestamp: {r['timestamp']}s | Mức độ khớp (Score): {r['score']:.4f}")