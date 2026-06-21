# AI EDGE Text-To-Speech

Dự án này chứa các giải pháp pipeline giọng nói thông minh trên thiết bị biên (Edge Voice Pipeline), đặc biệt được tối ưu hóa cho phần cứng nhúng như Raspberry Pi 5.

Dự án bao gồm 2 phiên bản chính, được chia thành 2 thư mục:

## 1. [AI_EDGE_S1](./AI_EDGE_S1) - Push-to-Talk Pipeline
Hệ thống nhận dạng và tổng hợp giọng nói offline 100% cho robot hình người.
* **Mô hình hoạt động:** Bấm nút → nói → nhả nút → robot phản hồi bằng giọng nói.
* **Đặc điểm:** Chạy hoàn toàn trên ARM CPU, không cần GPU hay internet, tối ưu cực thấp độ trễ (RTF < 0.3) với `whisper.cpp` (ASR) và `Piper` (TTS).

👉 [Xem chi tiết AI_EDGE_S1](./AI_EDGE_S1/README.md)

## 2. [AI_EDGE_S2](./AI_EDGE_S2) - Always-On Assistant
Hệ thống trợ lý giọng nói luôn lắng nghe (always-on) cho bảng điều khiển xe điện thông minh (EV Dashboard).
* **Mô hình hoạt động:** Microphone luôn mở, tích hợp VAD để lọc tiếng ồn môi trường, tự động nhận diện và phản hồi lệnh thoại hỗn hợp Anh-Việt.
* **Đặc điểm:** Quản lý tài nguyên CPU cực kỳ chặt chẽ (background ≤ 40%, active ≤ 70%), sử dụng `Silero VAD`, `SenseVoiceSmall` (ASR) và `Valtec-TTS` (TTS), hỗ trợ Normalizer chuyển đổi thuật ngữ kỹ thuật tiếng Anh.

👉 [Xem chi tiết AI_EDGE_S2](./AI_EDGE_S2/README.md)
# Always-on Voice Assistant — EV Dashboard on Raspberry Pi 5

Hệ thống trợ lý giọng nói **always-on offline** cho bảng điều khiển xe điện thông minh. Microphone luôn mở, VAD lọc tiếng ồn môi trường (gió, động cơ, còi xe), tự động nhận diện và phản hồi lệnh thoại bằng tiếng Việt — kể cả khi câu lệnh chứa thuật ngữ kỹ thuật tiếng Anh như "BMS overcurrent 24V" hay "CAN bus timeout".

**Vấn đề giải quyết**: trợ lý luôn lắng nghe nhưng không được chiếm CPU của motor controller và dashboard display. Hệ thống phải đọc các cảnh báo kỹ thuật hỗn hợp Anh-Việt với model TTS nhỏ (< 100M params) mà không làm phình to model hay tăng inference time.

---

## Công nghệ áp dụng

| Layer | Công nghệ | Lý do chọn |
|-------|-----------|------------|
| VAD | Silero VAD (ONNX, ~2MB) | < 5% CPU với 1 ONNX thread, state-of-the-art accuracy |
| ASR | SenseVoiceSmall (sherpa-onnx INT8, ~234M) | Non-autoregressive, 15× nhanh hơn Whisper-Large |
| TTS | Valtec-TTS (VITS2, ~74.8M params) | Zero-shot voice cloning, RTF ~0.24 trên CPU |
| Threading | Python threading + queue.Queue (bounded) | Producer-Consumer pattern, thread-safe, backpressure built-in |
| Code-switching | Regex/Rules Text Normalizer | 0.02ms/call, zero model overhead, domain dict dễ mở rộng |
| Prosody | VITS2 scalar params (length_scale, noise_scale) | 0% inference overhead, real-time adjustable |
| CPU control | psutil + adaptive exponential backoff | Đảm bảo budget ≤40% background / ≤70% active |
| Audio buffer | collections.deque(maxlen) Ring Buffer | O(1) discard, zero memory leak sau nhiều giờ |
| Hardware | Raspberry Pi 5 (BCM2712, 4× Cortex-A76) | ARMv8.2-a + NEON, LPDDR4X, share CPU với motor/dashboard |

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        RASPBERRY PI 5 — EV Dashboard                           │
│                                                                                 │
│  ┌─────────────────────────────────┐   ┌────────────────────────────────────┐  │
│  │  Thread 1: PRODUCER  (≤40% CPU) │   │  Thread 2: CONSUMER    (≤70% CPU)  │  │
│  │                                 │   │                                    │  │
│  │  Microphone (16kHz mono)        │   │  Queue.get(timeout=0.5s)           │  │
│  │       ↓                         │   │  [SLEEPS when queue empty]         │  │
│  │  Ring Buffer                    │   │       ↓                            │  │
│  │  deque(maxlen=93 chunks = 3s)   │   │  SenseVoice ASR (sherpa-onnx)     │  │
│  │       ↓ (luôn ghi)              │   │       ↓ text                       │  │
│  │  Silero VAD (ONNX, 1 thread)    │   │  CPU checkpoint (≤70%)             │  │
│  │  state: SILENCE→SPEECH→SILENCE  │   │       ↓                            │  │
│  │       ↓ (speech only)           │   │  CodeSwitch Normalizer             │  │
│  │  Pre-roll 300ms từ Ring Buffer  │   │  "BMS" → "bi em ét"               │  │
│  │       ↓                         │   │  "24V" → "24 vôn"                 │  │
│  │  [LỚP 1] VAD Timeout 10s        │   │       ↓                            │  │
│  │  [LỚP 2] Drop oldest @80% queue │   │  Severity Detect → Prosody Profile │  │
│  │  [LỚP 3] CPU adaptive backoff   │   │  normal/warning/critical           │  │
│  │       ↓                         │   │       ↓                            │  │
│  └─────────────┬───────────────────┘   │  Valtec-TTS (VITS2, 74.8M)        │  │
│                │                       │  length_scale / noise_scale        │  │
│                ▼                       │       ↓                            │  │
│     Thread-safe Queue                  │  Speaker (sounddevice)             │  │
│     maxsize=50, bounded                └────────────────────────────────────┘  │
│                │                                                                │
│                └──────────────────────────────►  Consumer thread               │
│                                                                                 │
│  Các tiến trình khác trên cùng CPU: motor controller, dashboard display        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Cấu trúc thư mục

```
AI_EDGE_S2/
│
├── main.py                             # Entry point
│                                       #   --demo: chạy text normalizer demo (không cần models)
│                                       #   (default): khởi động full pipeline
│
├── src/
│   ├── config.py                       # Tất cả constants tập trung
│   │                                   #   AudioConfig: 16kHz, 32ms chunk, 3s buffer
│   │                                   #   VADConfig: threshold 0.5/0.35, pre-roll 300ms
│   │                                   #   PipelineConfig: queue 50, timeout 10s, drop 80%
│   │                                   #   CPUConfig: bg ≤40%, active ≤70%
│   │
│   ├── audio/
│   │   └── ring_buffer.py              # Ring Buffer cố định chống memory leak
│   │                                   #   deque(maxlen=93) — tự discard chunk cũ O(1)
│   │                                   #   write(): ghi chunk 32ms, thread-safe
│   │                                   #   read_last_n_ms(): pre-roll audio trước VAD trigger
│   │
│   ├── vad/
│   │   ├── silero_vad.py               # Silero VAD engine (ONNX Runtime)
│   │   │                               #   4-state machine: SILENCE→PENDING→SPEECH→PENDING_SILENCE
│   │   │                               #   1 ONNX thread → CPU < 5% background
│   │   │                               #   min_speech_ms=250: lọc tiếng còi xe, gió ngắn
│   │   └── types.py                    # VADState enum, VADResult NamedTuple
│   │
│   ├── asr/
│   │   └── sensevoice.py               # SenseVoiceSmall wrapper (sherpa-onnx)
│   │                                   #   transcribe(): numpy float32 → text
│   │                                   #   num_threads=2, INT8 quantized (~60MB)
│   │
│   ├── nlp/
│   │   ├── dictionaries.py             # Từ điển domain xe điện
│   │   │                               #   ACRONYM_MAP: 48 entries (BMS, CAN, ECU, GPIO...)
│   │   │                               #   TECH_TERMS: 41 entries (overcurrent, timeout...)
│   │   │                               #   UNITS: 14 entries (V, A, °C, kWh, rpm...)
│   │   │                               #   LETTER_MAP: fallback spell-out A-Z → tiếng Việt
│   │   │
│   │   ├── normalizer.py               # Code-switching Text Normalizer
│   │   │                               #   Pipeline 4 tầng: Units→Acronyms→TechTerms→Fallback
│   │   │                               #   Pre-compile regex 1 lần → 0.02ms/call runtime
│   │   │                               #   Marker system tránh double-processing
│   │   │
│   │   └── severity.py                 # Severity Detector cho prosody tự động
│   │                                   #   detect(): "critical"/"warning"/"normal"
│   │                                   #   frozenset O(1) lookup, < 0.1ms/call
│   │
│   ├── tts/
│   │   ├── valtec_tts.py               # Valtec-TTS engine (VITS2)
│   │   │                               #   synthesize(): text → numpy PCM float32
│   │   │                               #   Tích hợp CodeSwitchNormalizer + ProsodyController
│   │   │
│   │   └── prosody.py                  # Prosody Controller — 3 scalar params VITS2
│   │                                   #   normal:   length_scale=1.0  (tốc độ bình thường)
│   │                                   #   warning:  length_scale=0.85 (+15% speed)
│   │                                   #   critical: length_scale=0.70 (+30% speed, pitch ổn định)
│   │                                   #   0% inference overhead (3 phép nhân scalar)
│   │
│   ├── system/
│   │   └── cpu_governor.py             # CPU Governor — adaptive throttle
│   │                                   #   Monitor thread đo CPU mỗi 1s (psutil)
│   │                                   #   throttle_if_needed(is_active): bg ≤40% / active ≤70%
│   │                                   #   Exponential backoff: 5ms→7.5ms→11ms→...max 50ms
│   │                                   #   Reset ngay về min khi CPU ổn
│   │
│   └── pipeline/
│       └── always_on.py                # AlwaysOnPipeline — orchestration chính
│                                       #   start(): khởi động 2 threads + CPU Governor
│                                       #   _producer_loop(): Mic→RingBuffer→VAD→Queue
│                                       #   _consumer_loop(): Queue→ASR→NLP→TTS→Speaker
│                                       #   3 lớp bảo vệ tràn: timeout/drop/throttle
│                                       #   Graceful shutdown: sentinel None + join
│
├── docs/
│   ├── 01_architecture.md              # Sơ đồ kiến trúc và lý do chọn model
│   ├── 02_implementation.md            # Pseudo-code và chi tiết implement
│   └── documentation_qa.md            # Q&A giải trình 3 câu hỏi thiết kế
│                                       #   (1) Code-switching: Regex vs Lexicon
│                                       #   (2) Prosody control 0% overhead
│                                       #   (3) Queue backpressure 3 lớp + UX impact
│
├── tests/
│   └── test_core.py                    # 63 test cases
│                                       #   Ring Buffer: memory leak, thread-safety, pre-roll
│                                       #   VAD: state machine, config values
│                                       #   Producer-Consumer: bounded queue, backpressure
│                                       #   CPU Governor: throttle, adaptive sleep, stats
│                                       #   Code-switching: acronyms, units, complex sentences
│                                       #   Performance: normalizer < 0.5ms, ring buffer < 1ms
│
├── models/                             # Model files (gitignored, download riêng)
│   ├── silero_vad.onnx                 # Silero VAD (~2MB)
│   ├── sensevoice-small/               # SenseVoiceSmall INT8 (~60MB)
│   └── valtec-tts/                     # Valtec-TTS VITS2 weights
│
├── requirements.txt                    # numpy, sounddevice, onnxruntime, sherpa-onnx, psutil
└── .gitignore
```

---

## Cài đặt & Chạy

```bash
# Cài dependencies
pip install -r requirements.txt

# Demo text normalizer — không cần models, chạy được ngay
python main.py --demo

# Full pipeline (cần models trong /models/)
python main.py

# Unit tests
python -m pytest tests/ -v
```

---

## Ràng buộc CPU và cách đáp ứng

| Chế độ | Giới hạn | Cơ chế |
|--------|---------|--------|
| Background (chỉ VAD đang nghe) | ≤ 40% CPU | Silero VAD 1 ONNX thread + adaptive sleep |
| Active (ASR + TTS đang inference) | ≤ 70% CPU | CPU checkpoint giữa ASR và TTS + exponential backoff |
| Queue overflow (tiếng ồn / nói quá dài) | Không treo | VAD Timeout 10s + Drop oldest @80% + bounded Queue 50 |

---

## Code-switching — Đọc câu hỗn hợp Anh-Việt

Model TTS (74.8M params) chỉ xử lý tiếng Việt. Text Normalizer chuyển đổi trước khi đưa vào model:

```
Input:  "Hệ thống đang kiểm tra BMS, phát hiện lỗi Overcurrent trên đường nguồn 24V"
           ↓ CodeSwitchNormalizer (0.02ms)
Output: "Hệ thống đang kiểm tra bi em ét, phát hiện lỗi ô-vơ-ca-rần trên đường nguồn 24 vôn"
           ↓ Valtec-TTS (VITS2, ~200ms)
[audio tiếng Việt tự nhiên]
```

Giải pháp này giữ model size không đổi (không nhúng từ điển Anh vào model), thêm từ mới chỉ cần edit `src/nlp/dictionaries.py`.
