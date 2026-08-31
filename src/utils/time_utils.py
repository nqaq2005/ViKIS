def format_seconds_to_hhmmss(seconds: float) -> str:
    """Chuyển đổi số giây (float) sang định dạng HH:MM:SS.mmm"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    
    # Lấy phần nghìn giây
    ms = int((seconds - int(seconds)) * 1000)
    
    if h > 0:
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"
    return f"{int(m):02d}:{int(s):02d}.{ms:03d}"

def timestamp_to_seconds(timestamp: str) -> float:
    """Chuyển đổi chuỗi HH:MM:SS hoặc MM:SS sang số giây."""
    parts = timestamp.strip().split(':')
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    else:
        return float(timestamp)

def frame_to_seconds(frame_index: int, fps: float) -> float:
    """Tính mốc thời gian (giây) dựa vào số thứ tự khung hình và FPS."""
    if fps <= 0:
        return 0.0
    return float(frame_index) / fps