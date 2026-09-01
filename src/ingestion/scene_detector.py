import os
import yaml
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from scenedetect import open_video, SceneManager, AdaptiveDetector

@dataclass
class ShotInfo:
    shot_id: int
    start: float       # Giây bắt đầu
    end: float         # Giây kết thúc
    duration: float    # Thời lượng (giây)
    start_frame: int
    end_frame: int

class SceneDetector:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        scene_cfg = config.get("ingestion", {}).get("scene_detection", {})
        self.adaptive_threshold = scene_cfg.get("adaptive_threshold", 3.0)
        self.min_scene_len_sec = scene_cfg.get("min_scene_len_sec", 1.0)

    def detect_scenes(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Phân đoạn video thành danh sách các Shot/Scene.
        
        :param video_path: Đường dẫn đến file video (.mp4, .mkv)
        :return: Danh sách các shot chứa thông tin timestamp và frame index
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy video tại: {video_path}")

        video = open_video(video_path)
        fps = video.frame_rate
        min_scene_len_frames = int(self.min_scene_len_sec * fps)

        scene_manager = SceneManager()
        scene_manager.add_detector(
            AdaptiveDetector(
                adaptive_threshold=self.adaptive_threshold,
                min_scene_len=min_scene_len_frames
            )
        )

        # Quét video và tìm các điểm chuyển cảnh
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        shots: List[ShotInfo] = []

        # Xử lý trường hợp không tìm thấy cảnh cắt nào (video ngắn hoặc tĩnh)
        if not scene_list:
            duration = video.duration
            if duration is None:
                raise ValueError("Không thể xác định thời lượng video.")
            
            total_frames = duration.get_frames()
            total_seconds = duration.get_seconds()
            shots.append(
                ShotInfo(
                    shot_id=1,
                    start=0.0,
                    end=round(total_seconds, 2),
                    duration=round(total_seconds, 2),
                    start_frame=0,
                    end_frame=total_frames
                )
            )
        else:
            for idx, (start_time, end_time) in enumerate(scene_list, start=1):
                start_sec = start_time.get_seconds()
                end_sec = end_time.get_seconds()
                shots.append(
                    ShotInfo(
                        shot_id=idx,
                        start=round(start_sec, 2),
                        end=round(end_sec, 2),
                        duration=round(end_sec - start_sec, 2),
                        start_frame=start_time.get_frames(),
                        end_frame=end_time.get_frames()
                    )
                )

        return [asdict(shot) for shot in shots]
