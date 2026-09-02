# ViKIS — Vietnamese Video Known-Item Search

**ViKIS** là hệ thống tìm kiếm sự kiện trong video (Known-Item Search) dành cho nội dung tiếng Việt. Hệ thống cho phép người dùng mô tả một sự kiện bằng ngôn ngữ tự nhiên và trả về video, khung hình cùng mốc thời gian `[t_start, t_end]` chứa sự kiện đó, dựa trên sự kết hợp giữa nội dung hình ảnh, lời thoại và văn bản xuất hiện trên màn hình.

## Tính năng chính

- **Tìm kiếm đa phương thức (Hybrid Search):** kết hợp tìm kiếm theo hình ảnh (text-to-image) và tìm kiếm theo lời thoại (dense + sparse), hợp nhất kết quả bằng Reciprocal Rank Fusion (RRF).
- **Pipeline nạp dữ liệu tự động:** phân đoạn cảnh (scene detection), trích xuất keyframe chất lượng cao, nhận dạng giọng nói tiếng Việt (ASR), và nhận dạng văn bản trong khung hình (OCR).
- **Tinh chỉnh kết quả theo thời gian:** hợp nhất kết quả visual và transcript theo cửa sổ thời gian (temporal fusion), loại bỏ trùng lặp bằng Non-Maximum Suppression (NMS), và quét mịn lại (coarse-to-fine decoding) để xác định thời điểm chính xác.
- **Tái xếp hạng bằng Cross-Encoder:** nâng cao độ chính xác kết quả văn bản với mô hình reranker chuyên biệt.
- **Giao diện web & API:** ứng dụng minh họa bằng Streamlit và backend API bằng FastAPI.

## Kiến trúc hệ thống

```
Video đầu vào
     │
     ▼
┌─────────────────────────── INGESTION ─────────────────────────────────┐
│  Scene Detection → Keyframe Extraction → ASR (giọng nói)              │
│                                        → OCR (văn bản trên hình)      │
└───────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────── EMBEDDING ─────────────────────────────────┐
│  Visual Encoder (Jina-CLIP v2)   │   Text Encoder (BGE-M3)            │
└───────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────── LƯU TRỮ (Qdrant) ─────────────────────────────┐
│  Collection: visual_keyframes    │   Collection: transcript_segments  │
└───────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────── RETRIEVAL ─────────────────────────────────┐
│  Hybrid Retriever → Temporal Fusion (RRF + NMS) → Fine Decoder        │
│                   → Reranker (Cross-Encoder)                          │
└───────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────── GENERATION ────────────────────────────────┐
│  Vision-Language Model tổng hợp câu trả lời + mốc thời gian           │
└───────────────────────────────────────────────────────────────────────┘
     │
     ▼
Kết quả: video, khung hình, [t_start, t_end], trích dẫn bằng chứng
```

## Công nghệ sử dụng

| Thành phần | Mô hình / Thư viện |
|---|---|
| Visual Embedding | `jinaai/jina-clip-v2` |
| Text Embedding (Hybrid Dense + Sparse) | `BAAI/bge-m3` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Nhận dạng giọng nói (ASR) | `hynt/Zipformer-30M-RNNT-6000h` (sherpa-onnx) |
| Nhận dạng văn bản (OCR) | EasyOCR (`vi`, `en`) |
| Phân đoạn cảnh | PySceneDetect (Adaptive Detector) |
| Vector Database | Qdrant |
| Backend API | FastAPI |
| Giao diện minh họa | Streamlit |



## Yêu cầu hệ thống

- Python 3.10 trở lên
- GPU hỗ trợ CUDA, VRAM: 12GB (khuyến nghị cho embedding và OCR; ASR có thể chạy trên CPU)
- Một instance Qdrant đang chạy (local hoặc cloud)

## Cài đặt

1. **Sao chép dự án và cài đặt thư viện:**

   ```bash
   git clone <repository-url>
   cd ViKIS
   pip install -r requirements.txt
   ```

2. **Cấu hình biến môi trường:**

   Sao chép file mẫu và điền thông tin kết nối Qdrant / Hugging Face:

   ```bash
   cp .env.example .env
   ```

   ```
   QDRANT_URL=your_qdrant_url_here
   QDRANT_API_KEY=your_qdrant_api_key_here
   QDRANT_TIMEOUT=60
   HF_TOKEN=your_huggingface_token_here
   ```

3. **Tùy chỉnh cấu hình (tùy chọn):**

   Điều chỉnh các tham số về phân đoạn cảnh, trích xuất keyframe, mô hình embedding và retrieval trong thư mục `configs/`.

## Sử dụng

### 1. Nạp và lập chỉ mục video

Đặt các file video (`.mp4`, `.mkv`, `.avi`, `.webm`, `.mov`) vào thư mục `data/raw_videos/`, sau đó chạy:

```bash
python scripts/run_indexing.py
```

Pipeline sẽ tự động phân đoạn cảnh, trích xuất keyframe, chạy ASR và OCR, sinh embedding và lưu vào Qdrant.

### 2. Khởi chạy giao diện tìm kiếm

```bash
streamlit run app/streamlit_app.py
```

### 3. Khởi chạy API backend

```bash
uvicorn app.api:app --reload
```

### 4. Đánh giá hệ thống

```bash
python scripts/evaluate_kis.py
```

## Giấy phép

Dự án được phát hành theo giấy phép [MIT License](LICENSE).