import yaml
from collections import defaultdict
from typing import List, Dict, Any, Tuple

class TemporalFusion:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        retrieval_cfg = self.config.get("retrieval", {})
        self.window_margin = retrieval_cfg.get("temporal_window_margin", 3.0)
        self.rrf_k = retrieval_cfg.get("rrf_k", 60)

    def _compute_rrf(self, hits: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """
        Tính điểm Reciprocal Rank Fusion (RRF) cho danh sách kết quả (đã được sắp xếp theo score giảm dần từ Qdrant).
        Công thức: RRF_Score = 1 / (k + Rank)
        """
        # Đảm bảo đã sort theo score giảm dần
        sorted_hits = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)
        
        for rank, item in enumerate(sorted_hits, start=1):
            item["rrf_score"] = 1.0 / (k + rank)
            item["rank"] = rank
            
        return sorted_hits

    def _nms_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Non-Maximum Suppression (NMS) theo thời gian.
        Loại bỏ các sự kiện trùng lặp/chồng lấp thời gian trên cùng 1 video, giữ lại sự kiện có điểm cao nhất.
        """
        # Sắp xếp các sự kiện theo điểm tổng hợp giảm dần
        sorted_events = sorted(events, key=lambda x: x["combined_score"], reverse=True)
        keep = []

        for event in sorted_events:
            is_overlap = False
            for k_event in keep:
                if event["video_id"] == k_event["video_id"]:
                    # Mở rộng các point-event (vd chỉ có 1 khung hình) thành khoảng tối thiểu 1 giây để dễ so sánh
                    e1_start = event["start_time"]
                    e1_end = max(event["start_time"] + 1.0, event["end_time"])
                    
                    e2_start = k_event["start_time"]
                    e2_end = max(k_event["start_time"] + 1.0, k_event["end_time"])
                    
                    # Tính toán khoảng thời gian giao nhau
                    overlap_start = max(e1_start, e2_start)
                    overlap_end = min(e1_end, e2_end)
                    
                    if overlap_start <= overlap_end:
                        is_overlap = True
                        # Tuỳ chọn: Có thể mở rộng ranh giới của k_event để bao phủ luôn event này
                        # k_event["start_time"] = min(k_event["start_time"], event["start_time"])
                        # k_event["end_time"] = max(k_event["end_time"], event["end_time"])
                        break
            
            if not is_overlap:
                keep.append(event)
                
        return keep

    def fuse(
        self,
        visual_hits: List[Dict[str, Any]],
        transcript_hits: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Hợp nhất 2 luồng kết quả Visual và Transcript dựa trên căn chỉnh thời gian (Temporal Alignment).
        """
        # 1. Tính RRF cho từng luồng
        v_hits = self._compute_rrf(visual_hits, self.rrf_k)
        t_hits = self._compute_rrf(transcript_hits, self.rrf_k)

        # 2. Gom nhóm theo Video ID
        vid_to_v = defaultdict(list)
        vid_to_t = defaultdict(list)
        all_vids = set()

        for v in v_hits:
            vid = v["video_id"]
            vid_to_v[vid].append(v)
            all_vids.add(vid)
            
        for t in t_hits:
            vid = t["video_id"]
            vid_to_t[vid].append(t)
            all_vids.add(vid)

        events = []

        # 3. Quét từng Video để ghép cặp (Alignment)
        for vid in all_vids:
            v_list = vid_to_v[vid]
            t_list = vid_to_t[vid]
            
            matched_t_indices = set()
            
            # a. Khớp Visual vào Transcript gần nhất
            for v in v_list:
                v_time = v["timestamp"]
                
                # Tìm các đoạn thoại nằm trong khoảng [v_time - margin, v_time + margin]
                matching_ts = [
                    (idx, t) for idx, t in enumerate(t_list)
                    if (t["start_time"] - self.window_margin) <= v_time <= (t["end_time"] + self.window_margin)
                ]
                
                if matching_ts:
                    # Nếu có nhiều transcript khớp, chọn cái có điểm RRF cao nhất
                    best_idx, best_t = max(matching_ts, key=lambda x: x[1]["rrf_score"])
                    matched_t_indices.add(best_idx)
                    
                    combined_score = v["rrf_score"] + best_t["rrf_score"]
                    events.append({
                        "video_id": vid,
                        "start_time": min(v_time, best_t["start_time"]),
                        "end_time": max(v_time, best_t["end_time"]),
                        "combined_score": combined_score,
                        "match_type": "multimodal_fusion",
                        "visual_info": v,
                        "transcript_info": best_t
                    })
                else:
                    # Visual không có transcript đi kèm
                    events.append({
                        "video_id": vid,
                        "start_time": v_time,
                        "end_time": v_time,
                        "combined_score": v["rrf_score"],
                        "match_type": "visual_only",
                        "visual_info": v,
                        "transcript_info": None
                    })
                    
            # b. Thêm các Transcript còn thừa (không khớp với bất kỳ Visual nào)
            for idx, t in enumerate(t_list):
                if idx not in matched_t_indices:
                    events.append({
                        "video_id": vid,
                        "start_time": t["start_time"],
                        "end_time": t["end_time"],
                        "combined_score": t["rrf_score"],
                        "match_type": "transcript_only",
                        "visual_info": None,
                        "transcript_info": t
                    })

        # 4. Lọc bỏ các khoảng thời gian bị chồng chéo (NMS) và trả về Top kết quả
        final_events = self._nms_events(events)
        return final_events


