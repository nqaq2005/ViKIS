import os
import cv2
import yaml
import numpy as np
import imagehash
from PIL import Image
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from decord import VideoReader, cpu

from src.utils.time_utils import frame_to_seconds

def _compute_frame_metrics(frame_rgb: np.ndarray) -> Tuple[float, float]:
    """Tính độ sắc nét (Laplacian) và độ sáng trung bình từ mảng RGB."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    return sharpness, brightness

def _process_video_worker(
    video_path: str,
    shots: List[Dict[str, Any]],
    output_dir: str,
    cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Hàm worker xử lý trích xuất keyframe cho 1 video độc lập."""
    if not os.path.exists(video_path):
        return []

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_dir, video_id)
    os.makedirs(save_dir, exist_ok=True)

    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    # Đọc tham số cấu hình
    short_thresh = cfg.get("short_shot_threshold", 8.0)
    sample_short = cfg.get("sample_rate_short", 0.5)
    sample_long = cfg.get("sample_rate_long", 2.0)
    safe_margin = cfg.get("safe_margin_sec", 0.3)
    min_sharpness = cfg.get("min_sharpness", 80.0)
    min_brightness = cfg.get("min_brightness", 25.0)
    phash_thresh = cfg.get("phash_threshold", 12)

    extracted_records = []

    for shot in shots:
        shot_id = shot["shot_id"]
        start_sec = shot["start"]
        end_sec = shot["end"]
        duration = end_sec - start_sec

        # Biên an toàn tránh nhiễu chuyển cảnh
        safe_start = start_sec + (safe_margin if duration > 1.0 else 0.0)
        safe_end = end_sec - (safe_margin if duration > 1.0 else 0.0)

        # -------------------------------------------------------------
        # TRƯỜNG HỢP 1: Shot ngắn (< 8s) -> Lấy mẫu dày & chọn 1 frame nét nhất
        # -------------------------------------------------------------
        if duration < short_thresh:
            sample_times = np.arange(safe_start, safe_end, sample_short)
            if len(sample_times) == 0:
                sample_times = [(start_sec + end_sec) / 2.0]

            frame_indices = [min(int(t * fps), total_frames - 1) for t in sample_times]
            frames = vr.get_batch(frame_indices).asnumpy()

            best_idx = -1
            max_sharpness = -1.0
            best_time = sample_times[0]

            for i, frame in enumerate(frames):
                sharpness, brightness = _compute_frame_metrics(frame)
                if brightness >= min_brightness and sharpness > max_sharpness:
                    max_sharpness = sharpness
                    best_idx = i
                    best_time = sample_times[i]

            # Fallback nếu mọi frame đều dưới ngưỡng chất lượng
            if best_idx == -1:
                best_idx = len(frames) // 2
                best_time = sample_times[best_idx]
                max_sharpness, _ = _compute_frame_metrics(frames[best_idx])

            frame_filename = f"shot_{shot_id:04d}_kf01.jpg"
            frame_path = os.path.join(save_dir, frame_filename)
            Image.fromarray(frames[best_idx]).save(frame_path, quality=92)

            frame_time = frame_to_seconds(frame_indices[best_idx], fps)
            extracted_records.append({
                "video_id": video_id,
                "scene_id": shot_id,
                "timestamp": round(float(frame_time), 2),
                "frame_path": frame_path,
                "sharpness": round(max_sharpness, 2)
            })

        # -------------------------------------------------------------
        # TRƯỜNG HỢP 2: Shot dài (>= 8s) -> Lấy mẫu thích ứng bằng pHash
        # -------------------------------------------------------------
        else:
            sample_times = np.arange(safe_start, safe_end, sample_long)
            frame_indices = [min(int(t * fps), total_frames - 1) for t in sample_times]
            frames = vr.get_batch(frame_indices).asnumpy()

            last_phash = None
            kf_counter = 1

            for i, frame in enumerate(frames):
                sharpness, brightness = _compute_frame_metrics(frame)
                if brightness < min_brightness or sharpness < min_sharpness:
                    continue

                pil_img = Image.fromarray(frame)
                curr_phash = imagehash.phash(pil_img)

                if last_phash is None or (curr_phash - last_phash) > phash_thresh:
                    frame_filename = f"shot_{shot_id:04d}_kf{kf_counter:02d}.jpg"
                    frame_path = os.path.join(save_dir, frame_filename)
                    pil_img.save(frame_path, quality=92)

                    frame_time = frame_to_seconds(frame_indices[i], fps)
                    extracted_records.append({
                        "video_id": video_id,
                        "scene_id": shot_id,
                        "timestamp": round(float(frame_time), 2),
                        "frame_path": frame_path,
                        "sharpness": round(sharpness, 2)
                    })

                    last_phash = curr_phash
                    kf_counter += 1

    return extracted_records


class KeyframeExtractor:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.full_config = yaml.safe_load(f)

        self.kf_cfg = self.full_config.get("ingestion", {}).get("keyframe_extraction", {})
        self.output_dir = self.full_config.get("paths", {}).get("keyframes_dir", "data/keyframes")
        self.num_workers = self.full_config.get("system", {}).get("num_workers", 4)
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_from_video(self, video_path: str, shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Xử lý đơn luồng cho 1 video."""
        return _process_video_worker(video_path, shots, self.output_dir, self.kf_cfg)

    def extract_batch(self, tasks: List[Tuple[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """Xử lý đa tiến trình hàng loạt video."""
        all_results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    _process_video_worker,
                    video_path,
                    shots,
                    self.output_dir,
                    self.kf_cfg
                ): video_path
                for video_path, shots in tasks
            }

            for future in as_completed(futures):
                v_path = futures[future]
                try:
                    res = future.result()
                    all_results.extend(res)
                    print(f"[EXTRACTOR] Hoàn thành {len(res)} frames từ: {os.path.basename(v_path)}")
                except Exception as e:
                    print(f"[EXTRACTOR ERROR] Lỗi khi xử lý {v_path}: {e}")

        return all_results