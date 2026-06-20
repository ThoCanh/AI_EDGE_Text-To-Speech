# PHẦN 3: GIẢI TRÌNH KỸ THUẬT (DOCUMENTATION) - AI EDGE S2

## Q1: Xử lý đa ngôn ngữ Code-switching

### Phương án chọn: Text-Normalization bằng Regex/Rules tại tầng phần mềm

**Chiến lược**: Chuyển đổi TẤT CẢ text tiếng Anh → phiên âm tiếng Việt (IPA Việt hóa) TRƯỚC khi đưa vào TTS model. Model TTS chỉ cần xử lý 1 ngôn ngữ duy nhất (tiếng Việt).

```
Input:  "Phát hiện lỗi Overcurrent trên đường nguồn 24V"
          ↓ Text Normalizer
Output: "Phát hiện lỗi ô-vơ ca-rần trên đường nguồn 24 vôn"
          ↓ Valtec-TTS (Vietnamese only)
Audio:  [natural Vietnamese pronunciation]
```

### So sánh 2 phương án

| Tiêu chí | Text-Norm (Regex/Rules) ✅ | Sửa Lexicon/Tokenizer ❌ |
|----------|---------------------------|-------------------------|
| **Tốc độ CPU** | ~0.1ms per sentence (regex) | Không thêm overhead nhưng cần retrain |
| **Model size** | Không thay đổi (74.8M) | Có thể tăng nếu thêm English phonemes |
| **Bảo trì** | Thêm từ mới = thêm 1 dòng dict | Cần retrain/fine-tune model |
| **Rủi ro** | Từ mới chưa có trong dict → đọc sai | Unstable nếu modify tokenizer |
| **Phù hợp Edge** | ✅ Zero overhead, deterministic | ❌ Tăng complexity, model size |

### Ưu điểm Text-Norm cho Edge
1. **CPU cost gần bằng 0**: Regex matching trên string ngắn = microseconds
2. **Deterministic**: Luôn cho kết quả giống nhau, không phụ thuộc model inference
3. **Extensible**: Domain-specific (xe điện) → từ điển nhỏ (~200 thuật ngữ), dễ bảo trì
4. **Không phình model**: Giữ nguyên 74.8M params, không cần thêm English embedding

### Nhược điểm
1. Từ tiếng Anh mới chưa có trong dictionary → fallback spell-out (đọc từng chữ cái)
2. Phiên âm có thể không hoàn hảo 100% (accent Việt hóa)
3. Cần maintain dictionary theo domain cụ thể

### Giải pháp fallback cho từ chưa biết
```python
def _fallback_spell_out(self, word: str) -> str:
    """Từ tiếng Anh không có trong dict → spell out từng chữ cái."""
    letter_map = {
        'a': 'ây', 'b': 'bi', 'c': 'xi', 'd': 'đi', 'e': 'i',
        'f': 'ép', 'g': 'gi', 'h': 'ết', 'i': 'ai', 'j': 'giây',
        'k': 'kây', 'l': 'eo', 'm': 'em', 'n': 'en', 'o': 'ô',
        'p': 'pi', 'q': 'kiu', 'r': 'a', 's': 'ét', 't': 'ti',
        'u': 'du', 'v': 'vi', 'w': 'đáp-liu', 'x': 'ích',
        'y': 'oai', 'z': 'dét',
    }
    return ' '.join(letter_map.get(c.lower(), c) for c in word)
```

---

## Q2: Kiểm soát Ngữ điệu (Prosody Control)

### Phương án: Parameter-level control TRƯỚC inference

Valtec-TTS (VITS2 architecture) cho phép điều chỉnh prosody qua các tham số đầu vào mà **KHÔNG cần thay đổi model** và **KHÔNG tăng inference time**.

### Các tham số điều chỉnh

```python
class ProsodyController:
    """Điều khiển ngữ điệu TTS theo ngữ cảnh cảnh báo."""
    
    PROFILES = {
        "normal": {
            "length_scale": 1.0,    # Tốc độ bình thường
            "noise_scale": 0.667,   # Biến thiên pitch bình thường
            "noise_scale_w": 0.8,   # Biến thiên duration
        },
        "warning": {
            "length_scale": 0.85,   # Nhanh hơn 15%
            "noise_scale": 0.5,     # Pitch ổn định hơn (nghiêm túc)
            "noise_scale_w": 0.6,   # Duration nhất quán
        },
        "critical": {
            "length_scale": 0.7,    # Nhanh hơn 30%
            "noise_scale": 0.3,     # Pitch rất ổn định (khẩn cấp)
            "noise_scale_w": 0.4,   # Duration rất nhất quán
        },
    }
    
    def get_params(self, severity: str) -> dict:
        return self.PROFILES.get(severity, self.PROFILES["normal"])
```

### Tại sao KHÔNG tăng inference time?

1. **length_scale**: Thay đổi tốc độ bằng cách scale duration prediction → cùng 1 forward pass, chỉ nhân/chia giá trị output
2. **noise_scale**: Điều chỉnh variance của latent noise → thay đổi giá trị input, không thêm computation
3. **Cả 3 params** đều là scalar multiplication trước/trong forward pass → O(1) overhead

### Phát hiện severity từ text

```python
CRITICAL_KEYWORDS = ["khẩn cấp", "critical", "nguy hiểm", "emergency", "overheat"]
WARNING_KEYWORDS = ["cảnh báo", "warning", "lỗi", "error", "fault"]

def detect_severity(text: str) -> str:
    text_lower = text.lower()
    if any(kw in text_lower for kw in CRITICAL_KEYWORDS):
        return "critical"
    elif any(kw in text_lower for kw in WARNING_KEYWORDS):
        return "warning"
    return "normal"
```

---

## Q3: Quản lý Hàng đợi & Backpressure

### Thiết kế 3 lớp bảo vệ

```
┌─────────────────────────────────────────────────┐
│  LỚP 1: VAD Timeout (10s max)                   │
│  → Force-cut speech sau 10 giây liên tục        │
│  → Reset VAD state machine                       │
├─────────────────────────────────────────────────┤
│  LỚP 2: Bounded Queue (maxsize=50)              │
│  → Queue.put(timeout=0.1) → non-blocking        │
│  → Khi đầy 80% → drop frame cũ nhất             │
├─────────────────────────────────────────────────┤
│  LỚP 3: CPU Throttle                            │
│  → Monitor CPU% → sleep adaptive khi vượt       │
│  → Background: sleep 10ms khi >40%              │
│  → Active: sleep 5ms khi >70%                   │
└─────────────────────────────────────────────────┘
```

### Chi tiết từng lớp

**Lớp 1 - VAD Timeout:**
```python
# Trong producer_thread:
if elapsed >= VAD_TIMEOUT_S:  # 10 seconds
    self._enqueue_speech(speech_chunks)  # Đẩy những gì có
    speech_chunks = []
    speech_start_time = None
    self.vad.reset()  # Reset VAD state
```
- **Mục đích**: Ngăn accumulate vô hạn khi môi trường ồn khiến VAD liên tục trigger
- **Tác dụng phụ**: Câu nói dài > 10s bị cắt đôi → ASR xử lý 2 phần riêng biệt

**Lớp 2 - Drop Frame (Backpressure):**
```python
def _enqueue_speech(self, chunks):
    # Khi queue > 80% capacity → drop oldest
    if self.audio_queue.qsize() >= int(QUEUE_MAXSIZE * 0.8):
        dropped = self.audio_queue.get_nowait()  # Bỏ frame cũ nhất
        del dropped
    
    # Non-blocking put
    try:
        self.audio_queue.put(audio, timeout=0.1)
    except queue.Full:
        del audio  # Drop nếu vẫn full
```
- **Mục đích**: Ưu tiên dữ liệu MỚI nhất (câu nói gần nhất quan trọng hơn)
- **Tác dụng phụ**: Mất 1-2 utterance cũ nếu hệ thống quá tải → người dùng có thể cần nhắc lại

**Lớp 3 - CPU Throttle:**
```python
def _throttle_producer(self):
    cpu = psutil.Process().cpu_percent()
    if consumer_busy and cpu > 70:
        time.sleep(0.005)   # Nhường CPU cho ASR/TTS
    elif not consumer_busy and cpu > 40:
        time.sleep(0.01)    # Nhường CPU cho motor/dashboard
```
- **Mục đích**: Đảm bảo luôn đủ CPU cho các tiến trình hệ thống khác
- **Tác dụng phụ**: Có thể miss speech chunk ngắn (~5-10ms) khi throttle

### Tác dụng phụ tổng thể lên UX

| Tình huống | Hành vi hệ thống | Ảnh hưởng UX |
|-----------|-------------------|--------------|
| Nói > 10s liên tục | Cắt + xử lý từng đoạn 10s | Phản hồi sớm hơn, nhưng mất ngữ cảnh liên tục |
| Ồn kéo dài | VAD trigger → timeout → reset | Có thể có vài phản hồi sai, sau đó tự ổn định |
| Queue tràn | Drop utterance cũ, giữ mới | Mất lệnh cũ, nhưng lệnh mới nhất luôn được xử lý |
| CPU cao | Throttle producer | Latency tăng nhẹ (~5-10ms), không đáng kể với UX |

### Kết luận
Thiết kế **"fail gracefully"**: Hệ thống KHÔNG BAO GIỜ treo hoặc crash. Worst case = mất 1-2 utterance cũ và cần người dùng nhắc lại. Đây là trade-off chấp nhận được cho embedded system 24/7.

---

## Timeline thực hiện

| Ngày | Task | Output |
|------|------|--------|
| 1 | Setup Pi5 + build sherpa-onnx + Silero VAD ONNX | Models chạy được |
| 2 | Implement RingBuffer + VAD Engine + Unit tests | Khối 1 hoàn chỉnh |
| 3 | Implement Producer-Consumer pipeline + Queue | Khối 2 hoàn chỉnh |
| 4 | Text Normalizer + TTS integration + Code-switch | Khối 3 hoàn chỉnh |
| 5 | Benchmark CPU%, backpressure test, documentation | Final submission |
