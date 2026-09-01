import os
import sys
import streamlit as st
from PIL import Image

# Đảm bảo import được các module trong src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.hybrid_retriever import VideoKISRetriever
from src.retrieval.temporal_fusion import TemporalFusion
from src.retrieval.fine_decoder import FineDecoder
from src.utils.time_utils import format_seconds_to_hhmmss
from decord import VideoReader, cpu


def get_fast_middle_frame(video_id: str, start_sec: float, end_sec: float, raw_dir: str = "data/raw_videos"):
    """Lấy nhanh 1 khung hình ở giữa đoạn audio để làm thumbnail, không qua AI."""
    extensions = [".mp4", ".mkv", ".avi", ".webm"]
    video_path = None
    for ext in extensions:
        p = os.path.join(raw_dir, f"{video_id}{ext}")
        if os.path.exists(p):
            video_path = p
            break

    if not video_path:
        return None

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        # Lấy mốc thời gian ở giữa đoạn audio
        mid_sec = (start_sec + end_sec) / 2.0
        frame_idx = min(int(mid_sec * fps), len(vr) - 1)

        frame_arr = vr[frame_idx].asnumpy()
        return Image.fromarray(frame_arr)
    except Exception:
        return None

# Cấu hình load mô hình (Chỉ load 1 lần duy nhất)
@st.cache_resource
def load_search_engine():
    retriever = VideoKISRetriever()
    fuser = TemporalFusion()
    # Dùng chung VisualEncoder từ retriever để tiết kiệm RAM & VRAM
    fine_decoder = FineDecoder(visual_encoder=retriever.visual_encoder)
    return retriever, fuser, fine_decoder



# GIAO DIỆN / STYLE (CSS tùy chỉnh)

CUSTOM_CSS = """
<style>
    /* Nền tổng thể và font */
    .stApp {
        background: radial-gradient(circle at top left, #131a2b 0%, #0b0f19 55%, #0a0d16 100%);
    }
    html, body, [class*="css"]  {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Header gradient */
    .kis-header {
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 45%, #db2777 100%);
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.25);
        margin-bottom: 1.4rem;
    }
    .kis-header h1 {
        color: white;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0;
    }
    .kis-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Khung tìm kiếm */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        backdrop-filter: blur(6px);
    }

    /* Nút bấm chính */
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.35);
    }
    .stFormSubmitButton > button[kind="primary"] {
    background-color: #2563eb !important;
    border-color: #2563eb !important;
    }
    .stFormSubmitButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
    }

    /* Thẻ kết quả */
    .result-card {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 16px;
        padding: 0.9rem 0.9rem 0.4rem 0.9rem;
        margin-bottom: 1rem;
        transition: border 0.15s ease-in-out;
    }
    .result-card:hover {
        border: 1px solid rgba(124, 58, 237, 0.55);
    }

    .rank-badge {
        display: inline-block;
        background: linear-gradient(120deg, #7c3aed, #db2777);
        color: white;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 2px 10px;
        border-radius: 999px;
        margin-right: 6px;
    }
    .score-badge {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        color: #e5e7eb;
        font-size: 0.78rem;
        padding: 2px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .meta-line {
        color: #9ca3af;
        font-size: 0.82rem;
        margin-top: 2px;
    }

    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }
</style>
"""

MATCH_LABELS = {
    "multimodal_fusion": ("✨ Khớp cả Hình & Tiếng", "success"),
    "visual_only": ("🖼️ Chỉ khớp Hình", "warning"),
    "transcript_only": ("🎤 Chỉ khớp Tiếng", "info"),
}


def main():
    st.set_page_config(page_title="Video KIS Search", page_icon="🔍", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown(
        """
        <div class="kis-header">
            <h1>KIS — Coarse-to-Fine Video Search</h1>
            <p>Tìm kiếm sự kiện trong kho video bằng hình ảnh &amp; lời thoại, kết hợp Reciprocal Rank Fusion.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Khởi tạo mô hình
    with st.spinner("Đang khởi tạo mô hình AI và kết nối Qdrant..."):
        retriever, fuser, fine_decoder = load_search_engine()

    if 'top_events' not in st.session_state:
        st.session_state['top_events'] = []
    if 'current_query' not in st.session_state:
        st.session_state['current_query'] = ""

    # Giao diện nhập liệu
    with st.form("search_form"):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            query = st.text_input("Nhập mô tả sự kiện (Tiếng Việt):", placeholder="Ví dụ: một nhân viên sở thú mặc áo xanh...")
        with col2:
            display_top_k = st.number_input("Lấy top:", min_value=1, max_value=100, value=4)
        with col3:
            retrieve_top_k = st.number_input("Độ sâu quét (Qdrant):", min_value=10, max_value=300, value=100)

        submit_search = st.form_submit_button("🔎 Tìm kiếm", type="primary", use_container_width=True)


    # 1. TẠO KHUNG CHỨA (PLACEHOLDER) TRỐNG

    result_container = st.empty()

    # Xử lý logic Tìm kiếm
    if submit_search and query:
        # LÀM SẠCH NGAY LẬP TỨC: Xóa dữ liệu cũ và các ảnh Quét mịn (Deep Scan) cũ
        st.session_state['top_events'] = []
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and key.startswith('fine_'):
                del st.session_state[key]

        # Dọn dẹp hẳn khu vực hiển thị cũ trên UI
        result_container.empty()

        # Bắt đầu vòng quay loading (lúc này UI phía dưới đã trắng tinh)
        with st.spinner("Đang quét trên Qdrant và tính toán RRF..."):
            raw_results = retriever.retrieve(query, retrieve_top_k=retrieve_top_k)

            fused_events = fuser.fuse(
                visual_hits=raw_results["visual"],
                transcript_hits=raw_results["transcript"]
            )

            st.session_state['top_events'] = fused_events[:display_top_k]
            st.session_state['current_query'] = query


    # 2. ĐƯA KẾT QUẢ MỚI VÀO KHUNG CHỨA

    with result_container.container():
        if st.session_state['top_events']:
            top_events = st.session_state['top_events']
            current_query = st.session_state['current_query']

            st.success(f"Đã hiển thị Top {len(top_events)} sự kiện cho: '{current_query}'")

            num_cols = 4
            cols = st.columns(num_cols)

            for idx, event in enumerate(top_events):
                col = cols[idx % num_cols]

                video_id = event["video_id"]
                match_type = event["match_type"]
                score = event["combined_score"]
                start_time = event["start_time"]
                end_time = event["end_time"]

                frame_path = event.get("visual_info", {}).get("frame_path") if event.get("visual_info") else None
                ocr_text = event.get("visual_info", {}).get("ocr_text", "") if event.get("visual_info") else ""
                transcript_text = event.get("transcript_info", {}).get("text", "") if event.get("transcript_info") else ""

                with col:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)

                    # 1. Hiển thị thông tin Thô (Coarse)
                    time_label = f"{format_seconds_to_hhmmss(start_time)} - {format_seconds_to_hhmmss(end_time)}"
                    st.markdown(
                        f'<span class="rank-badge">TOP {idx + 1}</span>'
                        f'<span class="score-badge">⭐ {score:.3f}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="meta-line">🎥 <code>{video_id}</code> &nbsp;|&nbsp; ⏱️ <code>{time_label}</code></div>',
                        unsafe_allow_html=True,
                    )
                    st.write("")

                    # Ưu tiên lấy ảnh từ fast_middle_frame cho audio, hoặc ảnh tĩnh Qdrant
                    if frame_path and os.path.exists(frame_path):
                        img = Image.open(frame_path)
                        st.image(img, use_container_width=True)
                    elif match_type == "transcript_only":
                        fast_img = get_fast_middle_frame(video_id, start_time, end_time)
                        if fast_img:
                            st.image(fast_img, caption="Ảnh trích xuất nhanh từ Audio", use_container_width=True)
                        else:
                            st.info("Không có Keyframe tĩnh")
                    else:
                        st.info("Không có Keyframe tĩnh")

                    fine_btn_key = f"btn_fine_{idx}"
                    if st.button("🔬 Quét Mịn (Deep Scan)", key=fine_btn_key, use_container_width=True):
                        with st.spinner("Trích xuất và chấm điểm frame trực tiếp..."):
                            fine_res = fine_decoder.refine_search(
                                video_id=video_id,
                                start_sec=start_time,
                                end_sec=end_time,
                                query_text=current_query,
                                top_k=3
                            )
                            st.session_state[f'fine_{idx}'] = fine_res

                    if f'fine_{idx}' in st.session_state:
                        fine_results = st.session_state[f'fine_{idx}']
                        if fine_results:
                            st.markdown("---")
                            st.markdown("🔥 **KẾT QUẢ DEEP SCAN:**")
                            for f_res in fine_results:
                                st.image(
                                    f_res["frame_image"],
                                    caption=f"⏱️ {f_res['timestamp']}s | 🎯 Score: {f_res['score']:.3f}",
                                    use_container_width=True
                                )
                        else:
                            st.warning("Lỗi trích xuất hoặc video gốc không tồn tại.")

                    with st.expander("📄 Chi tiết Text & Metadata"):
                        label_text, label_kind = MATCH_LABELS.get(match_type, ("Không xác định", "info"))
                        getattr(st, label_kind)(label_text)

                        if transcript_text:
                            st.markdown(f"**🎤 Lời thoại:** {transcript_text}")
                        if ocr_text:
                            st.markdown(f"**📝 OCR:** {ocr_text}")

                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()