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

def main():
    st.set_page_config(page_title="Video KIS Search", layout="wide")
    st.title("🔍 Hệ Thống KIS (Coarse-to-Fine Search)")
    
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
            query = st.text_input("Nhập mô tả sự kiện (Tiếng Việt):")
        with col2:
            display_top_k = st.number_input("Số lượng hiển thị:", min_value=1, max_value=50, value=5)
        with col3:
            retrieve_top_k = st.number_input("Độ sâu quét (Qdrant):", min_value=10, max_value=200, value=30)
            
        submit_search = st.form_submit_button("Tìm kiếm", type="primary")

    # ==========================================
    # 1. TẠO KHUNG CHỨA (PLACEHOLDER) TRỐNG 
    # ==========================================
    result_container = st.empty()

    # Xử lý logic Tìm kiếm
    if submit_search and query:
        # LÀM SẠCH NGAY LẬP TỨC: Xóa dữ liệu cũ và các ảnh Quét mịn (Deep Scan) cũ
        st.session_state['top_events'] = []
        for key in list(st.session_state.keys()):
            if key.startswith('fine_'):
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

    # ==========================================
    # 2. ĐỔ KẾT QUẢ MỚI VÀO KHUNG CHỨA
    # ==========================================
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
                    # 1. Hiển thị thông tin Thô (Coarse)
                    time_label = f"{format_seconds_to_hhmmss(start_time)} - {format_seconds_to_hhmmss(end_time)}"
                    st.markdown(f"**Top {idx + 1} | Điểm RRF: {score:.3f}**")
                    st.caption(f"🎥 `{video_id}` | ⏱️ `{time_label}`")

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
                    if st.button("🔍 Quét Mịn (Deep Scan)", key=fine_btn_key, use_container_width=True):
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
                            
                    with st.expander("Chi tiết Text & Metadata"):
                        if match_type == "multimodal_fusion":
                            st.success("Khớp cả Hình & Tiếng")
                        elif match_type == "visual_only":
                            st.warning("Chỉ khớp Hình")
                        else:
                            st.info("Chỉ khớp Tiếng")
                            
                        if transcript_text:
                            st.markdown(f"**🎤 Lời thoại:** {transcript_text}")
                        if ocr_text:
                            st.markdown(f"**📝 OCR:** {ocr_text}")
                    
                    st.markdown("---")

if __name__ == "__main__":
    main()