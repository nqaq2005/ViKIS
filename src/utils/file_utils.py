import os
import shutil
from typing import List

def ensure_dir(dir_path: str):
    """Đảm bảo thư mục tồn tại, nếu chưa có sẽ tự động tạo."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def get_all_videos(directory: str, extensions: tuple = (".mp4", ".mkv", ".avi", ".webm")) -> List[str]:
    """Lấy danh sách tất cả các đường dẫn file video trong một thư mục."""
    video_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                video_paths.append(os.path.join(root, file))
    return sorted(video_paths)

def clean_temp_directory(temp_dir: str):
    """Xóa an toàn thư mục tạm sau khi trích xuất hoặc xử lý xong."""
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"[FILE UTILS] Lỗi khi dọn dẹp thư mục tạm {temp_dir}: {e}")