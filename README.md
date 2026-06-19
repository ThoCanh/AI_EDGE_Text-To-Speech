# 🤖 Voice-to-Voice Pipeline (Push-to-Talk) — AI Edge on Raspberry Pi 5

> Bài kiểm tra năng lực AI Edge - Công ty MET  
> Module tương tác giọng nói **offline 100%** cho robot hình người

## Kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI 5                           │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │  Button   │──▶│  Audio Rec   │──▶│  ASR Engine  │        │
│  │  (GPIO)   │   │  PCM Buffer  │   │ whisper.cpp  │        │
│  └──────────┘   │  (in RAM)    │   │ Q5_0 GGUF    │        │
│                 └──────────────┘   └──────┬───────┘        │
│                                          │ text            │
│  ┌──────────┐   ┌──────────────┐   ┌─────▼────────┐        │
│  │  Speaker  │◀──│  /dev/shm/   │◀──│  TTS Engine  │        │
│  │  (ALSA)   │   │  (RAM Disk)  │   │  Piper TTS   │        │
│  └──────────┘   └──────────────┘   └──────────────┘        │
│                                                             │
│  Models loaded ONCE at boot → kept in RAM                   │
│  num_threads = 2 (optimal for Cortex-A76)                   │
└─────────────────────────────────────────────────────────────┘
```

## Cấu trúc thư mục

```
AI_EDGE/
├── src/
│   ├── voice_pipeline.py        # Main pipeline class
│   ├── asr_engine.py            # ASR wrapper (whisper.cpp)
│   ├── tts_engine.py            # TTS wrapper (Piper)
│   ├── audio_io.py              # Audio recording/playback
│   ├── memory_manager.py        # Memory monitoring & /dev/shm
│   └── config.py                # Hardware-specific constants
├── scripts/
│   ├── setup_pi5.sh             # Pi5 setup & build script
│   ├── quantize_whisper.sh      # Model quantization pipeline
│   └── benchmark.py             # RTF & memory benchmarks
├── docs/
│   ├── deep_dive_q5_quantization.md
│   ├── deep_dive_num_threads.md
│   └── deep_dive_dev_shm.md
├── tests/
│   └── test_pipeline.py         # Unit tests
├── models/                      # (gitignored) model files
├── requirements.txt
└── README.md
```

## Yêu cầu hệ thống

- **Hardware**: Raspberry Pi 5 (BCM2712, 4x Cortex-A76, 4GB+ RAM)
- **OS**: Raspberry Pi OS (64-bit ARM64) hoặc Ubuntu 24.04 ARM64
- **Python**: 3.11+
- **Build tools**: cmake, gcc/g++ (ARM64)

## Cài đặt nhanh trên Pi5

```bash
# 1. Clone project
git clone <repo_url> && cd AI_EDGE

# 2. Setup môi trường (build whisper.cpp + download models)
chmod +x scripts/setup_pi5.sh
./scripts/setup_pi5.sh

# 3. Chạy pipeline
python src/voice_pipeline.py
```

## KPIs

| Metric | Target | Achieved |
|--------|--------|----------|
| RTF (Real-Time Factor) | < 0.3 | Pending benchmark |
| Memory leak | None | ✅ Singleton + gc.collect() |
| Cold-start | 1 lần duy nhất | ✅ Warm-up at boot |
