# AI EDGE S2 - Always-on Voice Assistant with Code-switching TTS

> **Chủ đề**: Tối ưu hóa tài nguyên Always-on và Tái tạo giọng đọc tự nhiên, đa ngôn ngữ  
> **Target**: Raspberry Pi 5 (BCM2712, 4x Cortex-A76, 4-8GB RAM, Linux ARM64)

---

## Kiến trúc

```
Thread 1 (Producer ≤40% CPU)        Thread 2 (Consumer ≤70% CPU)
┌─────────────────────┐              ┌───────────────────────┐
│  Mic → RingBuffer   │              │  ASR (SenseVoice)     │
│  → Silero VAD       │──► Queue ──►│  → Text Normalizer    │
│  (3s, deque fixed)  │  (bounded)   │  → TTS (Valtec)      │
└─────────────────────┘              │  → Speaker             │
                                     └───────────────────────┘
```

## Cấu trúc dự án

```
AI_EDGE_S2/
├── main.py                          # Entry point
├── requirements.txt
├── src/
│   ├── config.py                    # Cấu hình tập trung
│   │
│   ├── audio/                       # 🎤 Audio I/O
│   │   ├── __init__.py
│   │   └── ring_buffer.py           # Ring Buffer (deque, fixed-size)
│   │
│   ├── vad/                         # 🔇 Voice Activity Detection
│   │   ├── __init__.py
│   │   └── silero_vad.py            # Silero VAD (4-state machine)
│   │
│   ├── asr/                         # 🧠 Speech Recognition
│   │   ├── __init__.py
│   │   └── sensevoice.py            # SenseVoiceSmall (sherpa-onnx)
│   │
│   ├── nlp/                         # 📝 Text Processing
│   │   ├── __init__.py
│   │   ├── dictionaries.py          # Từ điển domain (EV/automotive)
│   │   ├── normalizer.py            # Code-switching normalizer
│   │   └── severity.py              # Severity detector
│   │
│   ├── tts/                         # 🔊 Text-to-Speech
│   │   ├── __init__.py
│   │   ├── prosody.py               # Prosody controller
│   │   └── valtec_tts.py            # Valtec-TTS engine
│   │
│   ├── system/                      # ⚙️ System Monitoring
│   │   ├── __init__.py
│   │   └── cpu_governor.py          # CPU throttle (40%/70%)
│   │
│   └── pipeline/                    # 🔄 Orchestration
│       ├── __init__.py
│       └── always_on.py             # Producer-Consumer pipeline
│
├── docs/
│   ├── 01_architecture.md
│   ├── 02_implementation.md
│   └── 03_documentation.md
│
├── models/                          # Model files (gitignored)
└── tests/
    └── test_core.py
```

## Quick Start

```bash
# Demo text normalizer (không cần models)
python main.py --demo

# Chạy pipeline thật (cần models trong /models/)
python main.py

# Unit tests
python -m pytest tests/ -v
```

## Thiết kế module

| Module | Chức năng | File chính |
|--------|-----------|------------|
| `audio/` | Ring Buffer cố định, thu âm | `ring_buffer.py` |
| `vad/` | Lọc tiếng ồn, detect speech | `silero_vad.py` |
| `asr/` | Nhận dạng giọng nói | `sensevoice.py` |
| `nlp/` | Code-switching, severity | `normalizer.py`, `dictionaries.py` |
| `tts/` | Tổng hợp giọng + prosody | `valtec_tts.py`, `prosody.py` |
| `system/` | CPU monitoring + throttle | `cpu_governor.py` |
| `pipeline/` | Producer-Consumer orchestration | `always_on.py` |
