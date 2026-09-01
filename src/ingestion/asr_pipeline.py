import os
import json
import yaml
import tempfile
import subprocess
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import soundfile as sf
import sherpa_onnx
from huggingface_hub import hf_hub_download
import sentencepiece as spm


@dataclass
class TranscriptSegment:
    start: float       # Giây bắt đầu nói
    end: float         # Giây kết thúc nói
    text: str          # Nội dung câu thoại tiếng Việt


class ASRPipeline:
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        models_config_path: str = "configs/models_config.yaml",
        output_dir: str = ""
    ):
        # 1. Đọc cấu hình hệ thống và mô hình
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        with open(models_config_path, "r", encoding="utf-8") as f:
            self.models_config = yaml.safe_load(f)

        self.asr_cfg = self.models_config.get("asr_model", {})
        self.repo_id = self.asr_cfg.get("model_id", "hynt/Zipformer-30M-RNNT-6000h")
        self.sample_rate = self.asr_cfg.get("sample_rate", 16000)

        # Cấu hình VAD (Silero VAD qua sherpa-onnx)
        self.vad_cfg = self.models_config.get("vad_model", {})
        self.vad_filename = self.vad_cfg.get("filename", "silero_vad.onnx")
        self.vad_download_url = self.vad_cfg.get(
            "download_url",
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
        )
        self.vad_threshold = self.vad_cfg.get("threshold", 0.5)
        self.vad_min_silence_duration = self.vad_cfg.get("min_silence_duration", 0.3)
        self.vad_min_speech_duration = self.vad_cfg.get("min_speech_duration", 0.25)
        self.vad_max_speech_duration = self.vad_cfg.get("max_speech_duration", 20.0)

        if output_dir != "":
            self.transcripts_dir = output_dir
        else:
            self.transcripts_dir = self.config.get("paths", {}).get("transcripts_dir", "data/transcripts")
        os.makedirs(self.transcripts_dir, exist_ok=True)

        # 2. Chuẩn bị model weights và khởi tạo Recognizer + VAD
        self.model_dir = os.path.join("models", "zipformer_30m")
        self.vad_dir = os.path.join("models", "vad")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.vad_dir, exist_ok=True)

        self._ensure_model_downloaded()
        self._init_recognizer()
        self._ensure_vad_downloaded()
        self._init_vad()

  
    # Tải & khởi tạo model ASR

    def _ensure_model_downloaded(self):
        """Tải ONNX weights và bpe.model, sau đó trích xuất tokens.txt."""
        required_files = [
            "encoder-epoch-20-avg-10.onnx",
            "decoder-epoch-20-avg-10.onnx",
            "joiner-epoch-20-avg-10.onnx",
            "bpe.model"
        ]

        print(f"[ASR] Kiểm tra weights của {self.repo_id}...")
        for filename in required_files:
            file_path = os.path.join(self.model_dir, filename)
            if not os.path.exists(file_path):
                print(f"  └── Tải file '{filename}' từ HuggingFace Hub...")
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                    local_dir=self.model_dir,
                )

        # Tự động xuất tokens.txt từ bpe.model nếu chưa có
        tokens_path = os.path.join(self.model_dir, "tokens.txt")
        bpe_path = os.path.join(self.model_dir, "bpe.model")

        if not os.path.exists(tokens_path) and os.path.exists(bpe_path):
            print("[ASR] Đang tạo 'tokens.txt' từ 'bpe.model'...")
            sp = spm.SentencePieceProcessor()
            sp.load(bpe_path)

            with open(tokens_path, "w", encoding="utf-8") as f:
                for i in range(sp.get_piece_size()):
                    piece = sp.id_to_piece(i)
                    f.write(f"{piece} {i}\n")
            print(f"[ASR] Đã tạo thành công {tokens_path} ({sp.get_piece_size()} tokens).")

    def _init_recognizer(self):
        """Khởi tạo sherpa-onnx OfflineRecognizer cho Zipformer Transducer."""
        encoder_path = os.path.join(self.model_dir, "encoder-epoch-20-avg-10.onnx")
        decoder_path = os.path.join(self.model_dir, "decoder-epoch-20-avg-10.onnx")
        joiner_path = os.path.join(self.model_dir, "joiner-epoch-20-avg-10.onnx")
        tokens_path = os.path.join(self.model_dir, "tokens.txt")

        if not os.path.exists(tokens_path):
            raise FileNotFoundError(
                f"[ASR ERROR] Không tìm thấy '{tokens_path}'. "
                f"Kiểm tra lại bước tạo tokens.txt từ bpe.model."
            )

        print(f"[ASR] Khởi tạo Sherpa-ONNX Zipformer Recognizer (Sample Rate: {self.sample_rate}Hz)...")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            tokens=tokens_path,
            num_threads=2,
            sample_rate=self.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search"
        )

    # ------------------------------------------------------------------
    # Tải & khởi tạo VAD (Silero VAD)
    # ------------------------------------------------------------------
    def _ensure_vad_downloaded(self):
        """
        Tải model Silero VAD (onnx) nếu chưa có.
        Lưu ý: model này được k2-fsa host trên GitHub Releases,
        KHÔNG nằm trên HuggingFace Hub, nên phải tải trực tiếp qua URL
        thay vì dùng hf_hub_download.
        """
        vad_path = os.path.join(self.vad_dir, self.vad_filename)
        if os.path.exists(vad_path):
            return

        print(f"[VAD] Tải model VAD từ '{self.vad_download_url}'...")
        tmp_path = vad_path + ".tmp"
        try:
            urllib.request.urlretrieve(self.vad_download_url, tmp_path)
            os.replace(tmp_path, vad_path)
            print(f"[VAD] Đã tải xong: {vad_path}")
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(
                f"[VAD ERROR] Không thể tải model VAD từ '{self.vad_download_url}': {e}\n"
                f"Bạn có thể tải thủ công file này và đặt vào '{vad_path}'."
            ) from e

    def _init_vad(self):
        """Khởi tạo sherpa-onnx VoiceActivityDetector."""
        vad_path = os.path.join(self.vad_dir, self.vad_filename)

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = vad_path
        vad_config.silero_vad.threshold = self.vad_threshold
        vad_config.silero_vad.min_silence_duration = self.vad_min_silence_duration
        vad_config.silero_vad.min_speech_duration = self.vad_min_speech_duration
        vad_config.silero_vad.max_speech_duration = self.vad_max_speech_duration
        vad_config.sample_rate = self.sample_rate

        self.vad_config = vad_config
        print(
            f"[VAD] Đã khởi tạo Silero VAD "
            f"(threshold={self.vad_threshold}, max_speech={self.vad_max_speech_duration}s)."
        )

    def _new_vad_detector(self) -> "sherpa_onnx.VoiceActivityDetector":
        """Tạo instance VAD mới (mỗi lần transcribe dùng 1 instance sạch, tránh giữ state cũ)."""
        return sherpa_onnx.VoiceActivityDetector(self.vad_config, buffer_size_in_seconds=100)

  
    # Trích audio bằng FFmpeg
  
    def extract_audio(self, video_path: str, output_wav_path: str) -> bool:
        """Dùng FFmpeg tách audio từ video sang WAV Mono 16kHz."""
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",
            output_wav_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[ASR ERROR] Không thể tách âm thanh từ {video_path}: {e}")
            return False


    # Gom token thành câu (dùng cho fallback khi 1 đoạn VAD có timestamps chi tiết)

    def _segment_tokens(
        self,
        tokens: List[str],
        timestamps: List[float],
        silence_threshold_sec: float = 1.0,
        max_duration_sec: float = 12.0
    ) -> List[TranscriptSegment]:
        """
        Gom các tokens và timestamps từ Zipformer thành các đoạn câu thoại tự nhiên.
        - Tách câu khi khoảng lặng giữa 2 token liên tiếp > silence_threshold_sec.
        - Tách câu khi độ dài đoạn vượt quá max_duration_sec.
        """
        if not tokens or not timestamps:
            return []

        segments: List[TranscriptSegment] = []
        current_tokens = [tokens[0]]
        seg_start = timestamps[0]
        seg_end = timestamps[0]

        for i in range(1, len(tokens)):
            curr_token = tokens[i]
            curr_time = timestamps[i]
            prev_time = timestamps[i - 1]

            gap = curr_time - prev_time
            duration = curr_time - seg_start

            if gap > silence_threshold_sec or duration >= max_duration_sec:
                text = "".join(current_tokens).replace(" ", " ").strip().lower()
                if text:
                    segments.append(
                        TranscriptSegment(
                            start=round(float(seg_start), 2),
                            end=round(float(seg_end + 0.3), 2),
                            text=text
                        )
                    )
                current_tokens = [curr_token]
                seg_start = curr_time
            else:
                current_tokens.append(curr_token)

            seg_end = curr_time

        if current_tokens:
            text = "".join(current_tokens).replace(" ", " ").strip().lower()
            if text:
                segments.append(
                    TranscriptSegment(
                        start=round(float(seg_start), 2),
                        end=round(float(seg_end + 0.3), 2),
                        text=text
                    )
                )

        return segments

    # Decode audio dài bằng VAD (cắt nhỏ trước khi đưa vào offline recognizer)
  
    def _transcribe_with_vad(self, audio_samples, sr: int) -> List[TranscriptSegment]:
        """
        Dùng Silero VAD để cắt audio dài thành các đoạn có tiếng nói,
        sau đó decode từng đoạn riêng lẻ qua OfflineRecognizer.
        Giúp tránh việc nạp nguyên audio dài vào 1 lần decode (gây bùng nổ RAM).
        """
        vad = self._new_vad_detector()
        window_size = self.vad_config.silero_vad.window_size
        total_samples = len(audio_samples)

        segments: List[TranscriptSegment] = []
        i = 0

        def _flush_ready_segments():
            while not vad.empty():
                speech = vad.front
                vad.pop()

                start_time = speech.start / sr
                duration_sec = len(speech.samples) / sr

                stream = self.recognizer.create_stream()
                stream.accept_waveform(sr, speech.samples)
                self.recognizer.decode_stream(stream)
                result = stream.result

                text = ""
                if hasattr(result, "tokens") and hasattr(result, "timestamps") and result.tokens:
                    sub_segments = self._segment_tokens(result.tokens, result.timestamps)
                    if sub_segments:
                        for sub in sub_segments:
                            segments.append(
                                TranscriptSegment(
                                    start=round(start_time + sub.start, 2),
                                    end=round(start_time + sub.end, 2),
                                    text=sub.text.lower()
                                )
                            )
                        continue
                    text = result.text.strip().lower()
                else:
                    text = result.text.strip().lower()

                if text:
                    segments.append(
                        TranscriptSegment(
                            start=round(start_time, 2),
                            end=round(start_time + duration_sec, 2),
                            text=text
                        )
                    )

        # Đẩy audio vào VAD theo từng cửa sổ nhỏ
        while i < total_samples:
            chunk = audio_samples[i:i + window_size]
            if len(chunk) < window_size:
                pad = [0.0] * (window_size - len(chunk))
                chunk = list(chunk) + pad
            vad.accept_waveform(chunk)
            i += window_size
            _flush_ready_segments()

        # Báo hiệu hết audio, lấy nốt các đoạn còn dang dở trong buffer
        vad.flush()
        _flush_ready_segments()

        return segments

    # Hàm chính: transcribe video -> list transcript segments

    def transcribe(self, video_path: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Trích xuất toàn bộ transcript có gắn timestamp của video.
        Sử dụng VAD để cắt audio dài thành từng đoạn tiếng nói ngắn trước khi
        đưa vào OfflineRecognizer, tránh nạp nguyên audio dài gây bùng nổ RAM.

        :param video_path: Đường dẫn tới file video
        :param use_cache: Tận dụng file JSON đã lưu nếu đã transcribe trước đó
        :return: Danh sách dict [{'start': float, 'end': float, 'text': str}, ...]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

        video_id = os.path.splitext(os.path.basename(video_path))[0]
        json_cache_path = os.path.join(self.transcripts_dir, f"{video_id}.json")

        # 1. Đọc từ cache nếu có sẵn
        if use_cache and os.path.exists(json_cache_path):
            with open(json_cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 2. Tách audio ra file tạm
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_wav_path = temp_audio.name

        try:
            success = self.extract_audio(video_path, temp_wav_path)
            if not success:
                return []

            # 3. Đọc dữ liệu sóng âm
            audio_samples, sr = sf.read(temp_wav_path, dtype="float32")

            # 4. Cắt nhỏ bằng VAD rồi decode từng đoạn (thay vì decode nguyên file)
            segments = self._transcribe_with_vad(audio_samples, sr)

            # 4b. Fallback: nếu VAD không phát hiện được đoạn nói nào
            if not segments:
                print(f"[ASR WARNING] VAD không phát hiện tiếng nói nào trong '{video_id}'.")

        finally:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

        records = [asdict(seg) for seg in segments]

        # 5. Lưu cache kết quả dạng JSON
        with open(json_cache_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"[ASR] Đã nhận diện {len(records)} segments từ '{video_id}' -> lưu vào {json_cache_path}")
        return records
