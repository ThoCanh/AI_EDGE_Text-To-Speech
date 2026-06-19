# PHẦN 2: CHI TIẾT IMPLEMENTATION - AI EDGE S2

## KHỐI 1: Ring Buffer + VAD

### Ring Buffer Design

```python
import collections
import numpy as np
import threading

class RingBuffer:
    """
    Ring Buffer cố định cho audio stream.
    Dùng collections.deque(maxlen=N) → tự động discard frame cũ nhất.
    Tuyệt đối KHÔNG dùng list.append() vô hạn → memory leak.
    """
    
    SAMPLE_RATE = 16000
    CHUNK_MS = 32          # 32ms per chunk (Silero VAD optimal)
    BUFFER_SECONDS = 3     # Lưu tối đa 3 giây gần nhất
    
    def __init__(self):
        self.chunk_size = int(self.SAMPLE_RATE * self.CHUNK_MS / 1000)  # 512 samples
        max_chunks = int(self.BUFFER_SECONDS * 1000 / self.CHUNK_MS)   # ~94 chunks
        
        # deque(maxlen) → khi đầy, tự động pop phần tử cũ nhất
        self._buffer = collections.deque(maxlen=max_chunks)
        self._lock = threading.Lock()
    
    def write(self, chunk: np.ndarray):
        """Ghi chunk vào buffer. Thread-safe. O(1) amortized."""
        with self._lock:
            self._buffer.append(chunk.copy())
    
    def read_all(self) -> np.ndarray:
        """Đọc toàn bộ buffer thành 1 array liên tục."""
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self._buffer))
    
    def read_last_n_ms(self, ms: int) -> np.ndarray:
        """Đọc N ms gần nhất từ buffer (pre-roll)."""
        n_chunks = int(ms / self.CHUNK_MS)
        with self._lock:
            chunks = list(self._buffer)[-n_chunks:]
            if not chunks:
                return np.array([], dtype=np.float32)
            return np.concatenate(chunks)
    
    def clear(self):
        with self._lock:
            self._buffer.clear()
    
    @property
    def duration_ms(self) -> float:
        return len(self._buffer) * self.CHUNK_MS
```

### VAD Engine

```python
import onnxruntime as ort
import numpy as np

class SileroVADEngine:
    """
    Silero VAD wrapper - ONNX runtime.
    Chạy 1 thread ONNX → CPU < 5% trên Pi5.
    """
    
    THRESHOLD_ON = 0.5     # Speech onset
    THRESHOLD_OFF = 0.35   # Speech offset  
    MIN_SPEECH_MS = 250    # Tối thiểu 250ms mới tính là speech
    MAX_SILENCE_MS = 700   # Sau 700ms silence → end of utterance
    
    def __init__(self, model_path: str):
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1      # 1 thread cho VAD
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self._session = ort.InferenceSession(model_path, sess_opts)
        
        # VAD internal state
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._state = "SILENCE"   # SILENCE | PENDING_SPEECH | SPEECH | PENDING_SILENCE
        self._speech_counter = 0
        self._silence_counter = 0
    
    def process_chunk(self, chunk: np.ndarray) -> dict:
        """
        Xử lý 1 chunk 32ms.
        Returns: {"is_speech": bool, "state": str, "probability": float}
        """
        # ONNX inference
        ort_inputs = {
            "input": chunk.reshape(1, -1),
            "h": self._h,
            "c": self._c,
            "sr": np.array([16000], dtype=np.int64)
        }
        prob, self._h, self._c = self._session.run(None, ort_inputs)
        speech_prob = float(prob[0][0])
        
        # State machine
        is_speech = self._update_state(speech_prob)
        
        return {
            "is_speech": is_speech,
            "state": self._state,
            "probability": speech_prob
        }
    
    def _update_state(self, prob: float) -> bool:
        """State machine: SILENCE → PENDING → SPEECH → PENDING_SILENCE → SILENCE"""
        if self._state == "SILENCE":
            if prob >= self.THRESHOLD_ON:
                self._state = "PENDING_SPEECH"
                self._speech_counter = 32  # 1 chunk = 32ms
            return False
            
        elif self._state == "PENDING_SPEECH":
            if prob >= self.THRESHOLD_ON:
                self._speech_counter += 32
                if self._speech_counter >= self.MIN_SPEECH_MS:
                    self._state = "SPEECH"
                    return True
            else:
                self._state = "SILENCE"
                self._speech_counter = 0
            return False
            
        elif self._state == "SPEECH":
            if prob < self.THRESHOLD_OFF:
                self._state = "PENDING_SILENCE"
                self._silence_counter = 32
            return True
            
        elif self._state == "PENDING_SILENCE":
            if prob >= self.THRESHOLD_ON:
                self._state = "SPEECH"
                self._silence_counter = 0
                return True
            else:
                self._silence_counter += 32
                if self._silence_counter >= self.MAX_SILENCE_MS:
                    self._state = "SILENCE"
                    self._silence_counter = 0
                    return False
            return True
        
        return False
    
    def reset(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._state = "SILENCE"
```

---

## KHỐI 2: Producer-Consumer Pipeline

```python
import threading
import queue
import numpy as np
import sounddevice as sd
import time
import gc
import psutil
import os

class AlwaysOnPipeline:
    """
    Pipeline Always-on: Producer-Consumer pattern.
    
    Thread 1 (Producer): Mic → RingBuffer → VAD → Queue (khi có speech)
    Thread 2 (Consumer): Queue → ASR → Text-Norm → TTS → Speaker
    
    CPU Budget:
      - Background (T1 only): ≤ 40%
      - Active (T1 + T2):     ≤ 70%
    """
    
    SAMPLE_RATE = 16000
    CHUNK_MS = 32
    QUEUE_MAXSIZE = 50          # Bounded queue → backpressure
    VAD_TIMEOUT_S = 10          # Max speech duration trước khi force-cut
    DROP_THRESHOLD = 0.8        # Drop frames khi queue đầy 80%
    
    def __init__(self, vad_path: str, asr_path: str, tts_path: str):
        # ---- Models (load 1 lần) ----
        self.vad = SileroVADEngine(vad_path)
        self.asr = SenseVoiceASR(asr_path)      # sherpa-onnx wrapper
        self.tts = ValtecTTSEngine(tts_path)     # Valtec wrapper
        self.text_norm = CodeSwitchNormalizer()   # Text normalization
        
        # ---- Shared resources ----
        self.ring_buffer = RingBuffer()
        self.audio_queue = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        
        # ---- Control flags ----
        self._running = threading.Event()
        self._running.set()
        self._consumer_busy = threading.Event()  # Consumer đang xử lý?
        
        # ---- CPU monitoring ----
        self._process = psutil.Process(os.getpid())
    
    def start(self):
        """Khởi động pipeline."""
        t1 = threading.Thread(target=self._producer_thread, 
                              name="Producer-VAD", daemon=True)
        t2 = threading.Thread(target=self._consumer_thread, 
                              name="Consumer-ASR", daemon=True)
        t1.start()
        t2.start()
        print("[BOOT] ✅ Always-on pipeline started.")
        return t1, t2
    
    # ═══════════════════════════════════════
    # THREAD 1: PRODUCER (Audio + VAD)
    # ═══════════════════════════════════════
    
    def _producer_thread(self):
        """
        Vòng lặp thu âm liên tục.
        - Đọc chunk 32ms từ mic
        - Đưa qua VAD
        - Nếu speech=True → đẩy vào Queue (thread-safe)
        - Nếu queue đầy → DROP frame (backpressure)
        """
        chunk_size = int(self.SAMPLE_RATE * self.CHUNK_MS / 1000)
        speech_chunks = []           # Accumulate speech chunks
        speech_start_time = None
        
        stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=chunk_size,
        )
        stream.start()
        
        while self._running.is_set():
            # 1. Đọc audio chunk từ mic
            chunk, overflowed = stream.read(chunk_size)
            chunk = chunk.flatten()
            
            # 2. Ghi vào Ring Buffer (luôn ghi, kể cả silence)
            self.ring_buffer.write(chunk)
            
            # 3. VAD inference
            vad_result = self.vad.process_chunk(chunk)
            
            if vad_result["is_speech"]:
                # === SPEECH DETECTED ===
                if speech_start_time is None:
                    speech_start_time = time.monotonic()
                    # Prepend pre-roll (300ms trước speech) từ ring buffer
                    pre_roll = self.ring_buffer.read_last_n_ms(300)
                    speech_chunks.append(pre_roll)
                
                speech_chunks.append(chunk)
                
                # --- VAD Timeout: force-cut nếu nói quá lâu ---
                elapsed = time.monotonic() - speech_start_time
                if elapsed >= self.VAD_TIMEOUT_S:
                    self._enqueue_speech(speech_chunks)
                    speech_chunks = []
                    speech_start_time = None
                    self.vad.reset()
                    
            else:
                # === SILENCE / END OF SPEECH ===
                if speech_chunks:
                    # Có speech đã accumulate → đẩy vào queue
                    self._enqueue_speech(speech_chunks)
                    speech_chunks = []
                    speech_start_time = None
            
            # 4. CPU throttle nếu cần
            self._throttle_producer()
        
        stream.stop()
        stream.close()
    
    def _enqueue_speech(self, chunks: list):
        """
        Đẩy speech data vào Queue một cách an toàn.
        Nếu queue gần đầy → DROP frame cũ nhất (backpressure).
        """
        audio = np.concatenate(chunks)
        
        # Backpressure: nếu queue > 80% → drop frame cũ nhất
        if self.audio_queue.qsize() >= int(self.QUEUE_MAXSIZE * self.DROP_THRESHOLD):
            try:
                dropped = self.audio_queue.get_nowait()
                del dropped
                print("[WARN] ⚠️ Queue backpressure: dropped oldest frame")
            except queue.Empty:
                pass
        
        # Non-blocking put với timeout
        try:
            self.audio_queue.put(audio, timeout=0.1)
        except queue.Full:
            print("[WARN] ⚠️ Queue full: frame dropped")
            del audio
    
    def _throttle_producer(self):
        """Giới hạn CPU cho producer thread."""
        cpu_percent = self._process.cpu_percent(interval=None)
        if self._consumer_busy.is_set():
            # Active mode: T1 + T2 đang chạy → nhường CPU
            if cpu_percent > 70:
                time.sleep(0.005)  # 5ms sleep
        else:
            # Background mode: chỉ T1
            if cpu_percent > 40:
                time.sleep(0.01)   # 10ms sleep
    
    # ═══════════════════════════════════════
    # THREAD 2: CONSUMER (ASR + TTS)
    # ═══════════════════════════════════════
    
    def _consumer_thread(self):
        """
        Consumer: Chờ Queue có data → ASR → TTS.
        Dùng Queue.get(timeout=0.5) → KHÔNG block cứng.
        Thread này SLEEP khi queue trống (không tốn CPU).
        """
        while self._running.is_set():
            try:
                # Blocking get với timeout → thread ngủ khi queue trống
                audio_data = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue   # Quay lại chờ tiếp, không block hệ thống
            
            self._consumer_busy.set()
            
            try:
                # 1. ASR Inference
                raw_text = self.asr.transcribe(audio_data)
                if not raw_text.strip():
                    continue
                print(f"[ASR] 📝 '{raw_text}'")
                
                # 2. Text Normalization (Code-switching)
                normalized = self.text_norm.normalize(raw_text)
                print(f"[NORM] 🔄 '{normalized}'")
                
                # 3. TTS Synthesis
                tts_audio = self.tts.synthesize(normalized)
                
                # 4. Playback
                sd.play(tts_audio, samplerate=22050)
                sd.wait()
                
            except Exception as e:
                print(f"[ERROR] Consumer: {e}")
            finally:
                self._consumer_busy.clear()
                self.audio_queue.task_done()
                gc.collect()  # Cleanup intermediate buffers
    
    def stop(self):
        self._running.clear()
        print("[SHUTDOWN] 🔴 Pipeline stopped.")
```

---

## KHỐI 3: Code-switching Text Normalizer

```python
import re
from typing import List, Tuple

class CodeSwitchNormalizer:
    """
    Xử lý đa ngôn ngữ Anh-Việt tại tầng Text Normalization.
    
    Chiến lược: Regex/Rules-based → KHÔNG phình to model.
    Pipeline: Detect language segments → Normalize → Phonemize
    """
    
    # Từ điển viết tắt kỹ thuật → phiên âm tiếng Việt
    ACRONYM_MAP = {
        "BMS": "bi em ét",
        "CAN": "can",           # Đọc như từ, không spell out
        "ECU": "i xi du",
        "OBD": "ô bi đi",
        "ABS": "ây bi ét",
        "GPS": "gi pi ét",
        "LED": "lét",
        "USB": "du ét bi",
        "CPU": "xi pi du",
        "RAM": "ram",
        "SoC": "ét ô xi",
        "PCB": "pi xi bi",
        "OTA": "ô ti ây",
        "API": "ây pi ai",
        "IoT": "ai ô ti",
    }
    
    # Thuật ngữ kỹ thuật tiếng Anh → phiên âm Việt
    TECH_TERMS = {
        "overcurrent": "ô-vơ ca-rần",
        "timeout": "thai ao",
        "communication": "cơm-miu-ni-kây-sần",
        "voltage": "vôn-tít",
        "battery": "bát-tơ-ri",
        "firmware": "phơm-we",
        "software": "xóp-we",
        "hardware": "hát-we",
        "driver": "đrai-vơ",
        "sensor": "xen-xơ",
        "module": "mô-đun",
        "error": "e-rơ",
        "warning": "wo-ninh",
        "critical": "crít-ti-cồ",
        "shutdown": "sát-đao",
        "overheat": "ô-vơ hít",
        "controller": "cơn-trô-lơ",
    }
    
    # Đơn vị đo
    UNITS = {
        "V": "vôn",
        "A": "am-pe",
        "W": "oát",
        "kW": "ki-lô oát",
        "mA": "mi-li am-pe",
        "°C": "độ xê",
        "Hz": "héc",
        "kHz": "ki-lô héc",
        "MHz": "mê-ga héc",
        "Ω": "ôm",
        "rpm": "vòng trên phút",
    }
    
    def normalize(self, text: str) -> str:
        """
        Pipeline normalization:
        1. Normalize số + đơn vị (24V → 24 vôn)
        2. Expand acronyms (BMS → bi em ét)
        3. Transliterate tech terms (Overcurrent → ô-vơ ca-rần)
        4. Clean up spacing
        """
        result = text
        
        # Step 1: Số + đơn vị
        result = self._normalize_units(result)
        
        # Step 2: Acronyms (all caps words)
        result = self._expand_acronyms(result)
        
        # Step 3: English tech terms
        result = self._transliterate_english(result)
        
        # Step 4: Cleanup
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def _normalize_units(self, text: str) -> str:
        """24V → 24 vôn, 100mA → 100 mi-li am-pe"""
        for unit, vn_name in sorted(self.UNITS.items(), 
                                      key=lambda x: -len(x[0])):
            pattern = rf'(\d+\.?\d*)\s*{re.escape(unit)}\b'
            text = re.sub(pattern, rf'\1 {vn_name}', text)
        return text
    
    def _expand_acronyms(self, text: str) -> str:
        """BMS → bi em ét"""
        for acronym, pronunciation in self.ACRONYM_MAP.items():
            text = re.sub(rf'\b{acronym}\b', pronunciation, text, 
                         flags=re.IGNORECASE)
        return text
    
    def _transliterate_english(self, text: str) -> str:
        """Overcurrent → ô-vơ ca-rần"""
        for eng, vn in self.TECH_TERMS.items():
            text = re.sub(rf'\b{eng}\b', vn, text, flags=re.IGNORECASE)
        return text


# === Ví dụ test ===
if __name__ == "__main__":
    norm = CodeSwitchNormalizer()
    
    tests = [
        "Hệ thống đang kiểm tra BMS, phát hiện lỗi Overcurrent trên đường nguồn 24V",
        "Mã lỗi CAN bus communication timeout",
        "Cảnh báo: Battery overheat, nhiệt độ 85°C",
        "Firmware OTA update thất bại, ECU không phản hồi",
    ]
    
    for t in tests:
        print(f"IN:  {t}")
        print(f"OUT: {norm.normalize(t)}")
        print()
```
