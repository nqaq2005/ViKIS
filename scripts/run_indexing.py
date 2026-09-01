import os
import time
from dotenv import load_dotenv
import yaml

from src.utils.file_utils import ensure_dir, get_all_videos
from src.ingestion.scene_detector import SceneDetector
from src.ingestion.keyframe_extractor import KeyframeExtractor
from src.ingestion.asr_pipeline import ASRPipeline
from src.ingestion.ocr_scanner import OCRScanner
from src.embeddings.visual_encoder import VisualEncoder
from src.embeddings.text_encoder import TextEncoder
from src.storage.init_qdrant import QdrantInitializer
from src.storage.qdrant_client import VideoKISQdrantClient


load_dotenv()


def run_pipeline(recreate_db: bool = False):
    start_total_time = time.time()
    
    # 1. Đọc cấu hình chung
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_videos_dir = config.get("paths", {}).get("raw_videos_dir", "data/raw_videos")
    ensure_dir(raw_videos_dir)
    video_paths = get_all_videos(raw_videos_dir)

    if not video_paths:
        print(f"[ABORT] Không tìm thấy video nào trong '{raw_videos_dir}'. Hãy thêm video vào thư mục này.")
        return

    print(f"=== BẮT ĐẦU PIPELINE INDEXING ({len(video_paths)} VIDEOS) ===")

    # 2. Khởi tạo Database Collections
    db_init = QdrantInitializer()
    db_init.initialize_collections(recreate=recreate_db)
    q_client = VideoKISQdrantClient()

    # 3. Nạp sẵn các mô hình AI vào bộ nhớ
    print("\n--- Khởi tạo các mô hình AI và Pipeline ---")
    scene_detector = SceneDetector()
    keyframe_extractor = KeyframeExtractor()
    asr_pipeline = ASRPipeline()
    ocr_scanner = OCRScanner()
    visual_encoder = VisualEncoder()
    text_encoder = TextEncoder()
    print("--- Khởi tạo hoàn tất! Bắt đầu xử lý từng video ---\n")

    # 4. Lặp qua từng video và xử lý
    for v_idx, v_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(v_path)
        video_id = os.path.splitext(video_name)[0]
        v_start_time = time.time()
        print(f"\n[{v_idx}/{len(video_paths)}] Đang xử lý: {video_name} (ID: {video_id})")

        # BƯỚC 1: Phân đoạn Shot/Scene (PySceneDetect)
        print("  [1/6] Phân đoạn cảnh (Scene Detection)...")
        shots = scene_detector.detect_scenes(v_path)
        print(f"       -> Phát hiện {len(shots)} shots.")

        # BƯỚC 2: Trích xuất Keyframes (Decord + Laplacian + pHash)
        print("  [2/6] Trích xuất Keyframes thích ứng...")
        keyframes = keyframe_extractor.extract_from_video(v_path, shots)
        print(f"       -> Đã trích xuất {len(keyframes)} keyframes.")

        # BƯỚC 3: Quét OCR trên Keyframes (EasyOCR)
        print("  [3/6] Quét chữ trên màn hình (Video OCR)...")
        keyframes = ocr_scanner.process_keyframes(video_id, keyframes, use_cache=True)

        # BƯỚC 4: Chuyển đổi giọng nói ASR (Zipformer RNN-T)
        print("  [4/6] Nhận diện lời thoại (ASR Pipeline)...")
        transcripts = asr_pipeline.transcribe(v_path)
        print(f"       -> Đã nhận diện {len(transcripts)} đoạn hội thoại.")

        # BƯỚC 5: Sinh Visual Embeddings & Upsert visual_keyframes
        if keyframes:
            print("  [5/6] Sinh Vector Jina-CLIP-v2 và nạp vào visual_keyframes...")
            image_paths = [kf["frame_path"] for kf in keyframes]
            visual_vectors = visual_encoder.encode_images(image_paths)
            q_client.upsert_keyframes(keyframes, visual_vectors)
        else:
            print("  [5/6] Bỏ qua visual (Không có keyframes hợp lệ).")

        # BƯỚC 6: Sinh Text Embeddings (BGE-M3 Dense + Sparse) & Upsert transcript_segments
        if transcripts:
            print("  [6/6] Sinh Vector BGE-M3 (Hybrid) và nạp vào transcript_segments...")
            transcript_texts = [tr["text"] for tr in transcripts]
            text_embeddings = text_encoder.encode_documents(transcript_texts)
            q_client.upsert_transcripts(video_id, transcripts, text_embeddings)
        else:
            print("  [6/6] Bỏ qua transcript (Video không có lời thoại).")

        v_duration = round(time.time() - v_start_time, 2)
        print(f"  [XONG] Video '{video_name}' hoàn thành trong {v_duration}s.")

    total_duration = round(time.time() - start_total_time, 2)
    print(f"\n=======================================================")
    print(f" TẤT CẢ DỮ LIỆU ĐÃ ĐƯỢC NẠP LÊN QDRANT THÀNH CÔNG!")
    print(f" Tổng thời gian chạy: {total_duration}s")
    print(f"=======================================================\n")

if __name__ == "__main__":
    # Đặt recreate_db=True nếu muốn xóa làm mới dữ liệu từ đầu
    run_pipeline(recreate_db=False)