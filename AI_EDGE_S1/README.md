# Voice-to-Voice Pipeline — Push-to-Talk on Raspberry Pi 5

Hệ thống nhận dạng và tổng hợp giọng nói **offline 100%** cho robot hình người. Người dùng bấm nút → nói → nhả nút → robot phản hồi bằng giọng nói, toàn bộ chạy trên ARM CPU không cần GPU hay kết nối mạng.

**Vấn đề giải quyết**: Các giải pháp ASR/TTS thương mại yêu cầu cloud, gây trễ 500ms–2s và không hoạt động offline. Hệ thống này đạt RTF < 0.3 (5 giây audio xử lý dưới 1.5s) hoàn toàn on-device trên phần cứng nhúng.

---

## Công nghệ áp dụng

| Layer | Công nghệ | Lý do chọn |
|-------|-----------|------------|
| ASR | whisper.cpp + Whisper-Tiny Q5_0 GGUF | GGML native, tự tune ARM NEON kernel, không dependency |
| TTS | Piper TTS (Vietnamese VITS) | Pre-built ARM64 binary, latency thấp, chạy persistent subprocess |
| Audio I/O | sounddevice (PortAudio) | Cross-platform, callback-based, zero-copy PCM |
| Inference | ONNX Runtime / whisper.cpp C++ | ARM NEON SIMD + dotprod acceleration tự động |
| Memory | /dev/shm (tmpfs) + gc.collect | RAM-only I/O, không hao mòn MicroSD |
| Quantization | GGUF Q5_0 (~30MB) | Sweet spot: WER delta < 2%, fit cache ARM A76 tốt hơn Q8 |
| Hardware | Raspberry Pi 5 (BCM2712, 4× Cortex-A76) | ARMv8.2-a + NEON + dotprod, LPDDR4X 34 GB/s |

---

## Kiến trúc hệ thống

```
┌──────────────────────────────── RASPBERRY PI 5 ─────────────────────────────────┐
│                                                                                  │
│  [GPIO Button]                                                                   │
│       │ Press                                                                    │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────┐                                    │
│  │           VoicePipeline                  │                                    │
│  │                                          │                                    │
│  │  BOOT (1 lần duy nhất):                 │                                    │
│  │    ASR model → RAM (whisper.cpp)         │                                    │
│  │    TTS model → RAM (Piper subprocess)    │                                    │
│  │    Warmup inference (prime CPU cache)    │                                    │
│  │                                          │                                    │
│  │  PUSH:                                   │                                    │
│  │    Mic → PCM float32 16kHz → bytearray  │                                    │
│  │                    (in RAM, no file I/O) │                                    │
│  │                                          │                                    │
│  │  RELEASE:                                │                                    │
│  │    bytearray ──► ASR (whisper.cpp Q5_0) │                                    │
│  │                      ▼ text             │                                    │
│  │                  TTS (Piper) ──► PCM    │                                    │
│  │                      ▼                  │                                    │
│  │    [/dev/shm/ optional] ──► Speaker     │                                    │
│  └─────────────────────────────────────────┘                                    │
│                                                                                  │
│  CPU budget:  2 threads inference  +  2 threads OS/audio/TTS                   │
│  Memory:      ~50MB ASR  +  ~80MB TTS  =  ~130MB total                         │
│  RTF target:  < 0.3  (5s audio → processed in < 1.5s)                          │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Cấu trúc thư mục

```
AI_EDGE_S1/
│
├── src/                            # Source code chính
│   ├── voice_pipeline.py           # Entry point — class VoicePipeline
│   │                               #   __init__(): load model 1 lần (warm-up)
│   │                               #   start_recording(): push event handler
│   │                               #   stop_and_process(): release → ASR → TTS → play
│   │
│   ├── asr_engine.py               # ASR wrapper cho whisper.cpp
│   │                               #   Load GGUF model qua pywhispercpp hoặc ctypes
│   │                               #   Zero-copy: truyền numpy pointer trực tiếp vào C
│   │                               #   warmup(): chạy dummy 1s để prime cache
│   │
│   ├── tts_engine.py               # TTS wrapper cho Piper
│   │                               #   Persistent subprocess (model load 1 lần)
│   │                               #   Giao tiếp stdin/stdout pipe (RAM buffer)
│   │                               #   Hỗ trợ /dev/shm/ khi cần chia sẻ file
│   │
│   ├── audio_io.py                 # Audio recording và playback
│   │                               #   AudioRecorder: sounddevice callback → bytearray RAM
│   │                               #   AudioPlayer: play numpy array qua ALSA/PortAudio
│   │
│   ├── noise_filter.py             # Lọc tạp âm DSP (không cần ML)
│   │                               #   Tầng 1: High-pass IIR filter (< 80Hz → bỏ tiếng ù/gió)
│   │                               #   Tầng 2: Noise gate theo RMS frame 20ms
│   │                               #   Tầng 3: Spectral gate STFT (tùy chọn, tắt mặc định)
│   │                               #   Cost: ~0.5ms cho 5s audio
│   │
│   ├── memory_manager.py           # Memory monitoring và /dev/shm management
│   │                               #   Theo dõi RSS process qua psutil
│   │                               #   Cảnh báo khi RSS tăng > 10MB (memory leak)
│   │                               #   Quản lý temp files trên tmpfs
│   │
│   └── config.py                   # Hardware constants tập trung
│                                   #   NUM_INFERENCE_THREADS = 2 (optimal Pi5)
│                                   #   TMPFS_DIR = /dev/shm (fallback tempdir nếu Windows)
│                                   #   TARGET_RTF = 0.3
│
├── scripts/
│   ├── quantize_whisper.sh         # Build whisper.cpp + quantize Whisper-Tiny → Q5_0
│   │                               #   cmake với -march=armv8.2-a+dotprod -O3
│   │                               #   Output: models/ggml-tiny-q5_0.bin (~30MB)
│   │
│   ├── setup_pi5.sh                # Full setup script cho Raspberry Pi 5
│   │                               #   Cài dependencies, build whisper.cpp, download models
│   │
│   └── benchmark.py                # Benchmark RTF và memory usage
│
├── docs/
│   ├── documentation_qa.md         # Q&A giải trình 4 câu hỏi thiết kế
│   │                               #   (1) whisper.cpp vs sherpa-onnx
│   │                               #   (2) num_threads = 2, lý do không dùng 4
│   │                               #   (3) /dev/shm/ thay vì /tmp/
│   │                               #   (4) Q5_0 thay vì FP16/INT8/Q8/Q4
│   │
│   ├── deep_dive_q5_quantization.md   # Phân tích chi tiết cache hierarchy A76
│   │                                  # và lý do Q5 nhanh hơn Q8 trên ARM
│   │
│   ├── deep_dive_num_threads.md    # Memory bandwidth saturation, cache thrashing,
│   │                               # fork-join overhead — tại sao num_threads=2 tối ưu
│   │
│   └── deep_dive_dev_shm.md        # /dev/shm vs /tmp vs /home — tmpfs trên Linux,
│                                   # wear leveling MicroSD, POSIX guarantee
│
├── tests/
│   └── test_pipeline.py            # Unit tests cho các module
│
├── models/                         # Model files (gitignored, download bằng script)
│   ├── ggml-tiny-q5_0.bin          # Whisper-Tiny Q5_0 (~30MB)
│   └── vi_VN-vais1000-medium.onnx  # Piper Vietnamese TTS voice
│
├── requirements.txt
└── .gitignore
```

---

## Cài đặt

### Trên Raspberry Pi 5

```bash
# 1. Clone và vào thư mục
git clone <repo> && cd AI_EDGE_S1

# 2. Cài Python dependencies
pip install -r requirements.txt

# 3. Build whisper.cpp và quantize model
chmod +x scripts/quantize_whisper.sh
./scripts/quantize_whisper.sh

# 4. Download Piper TTS Vietnamese voice
# (scripts/setup_pi5.sh sẽ tự download)
chmod +x scripts/setup_pi5.sh
./scripts/setup_pi5.sh
```

### Chạy pipeline

```bash
# Keyboard simulation (Enter = bấm/nhả nút)
python src/voice_pipeline.py

# GPIO mode (nút vật lý Pi5 GPIO17)
USE_GPIO=1 python src/voice_pipeline.py
```

### Chạy tests

```bash
python -m pytest tests/ -v
```

---

## KPIs

| Metric | Yêu cầu | Giải pháp |
|--------|---------|-----------|
| RTF < 0.3 | 5s audio → xử lý < 1.5s | Q5_0 ~30MB + num_threads=2 + ARM NEON |
| No memory leak | RSS ổn định sau nhiều lần dùng | Singleton model + del buffer + gc.collect() |
| Cold-start latency = 0 | Model không load lại mỗi lần bấm | Load 1 lần trong `__init__()`, warmup inference |
| Offline 100% | Không cần internet | whisper.cpp + Piper đều local |

---

## Quyết định kỹ thuật quan trọng

**whisper.cpp thay vì sherpa-onnx**: whisper.cpp viết riêng cho Whisper, tune từng GEMM kernel ARM NEON. Zero dependency (pure C++), GGUF quantization native. sherpa-onnx phù hợp hơn khi cần chạy nhiều model khác nhau trong cùng app.

**Q5_0 thay vì Q8_0**: Inference AI trên ARM là memory-bound (60% thời gian = fetch weights). Model nhỏ hơn (30MB vs 42MB) → ít bytes fetch → ít cache miss → nhanh hơn 20%, dù dequantize phức tạp hơn. WER delta ~1.5% vẫn trong ngưỡng 2%.

**num_threads = 2 thay vì 4**: L3 cache 2MB chia 4 thread = 512KB/thread → thrashing. Memory bandwidth LPDDR4X 34 GB/s bão hòa ở 2-3 threads. 2 core còn lại nhường cho OS, ALSA audio callback, TTS subprocess.

**Chi tiết đầy đủ**: xem thư mục [docs/](docs/).
