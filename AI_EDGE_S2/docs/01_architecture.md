# PHẦN 1: KIẾN TRÚC TỔNG THỂ - AI EDGE S2

## Tổng quan hệ thống

| Hạng mục | Chi tiết |
|----------|----------|
| **Chủ đề** | Always-on Voice Assistant + Code-switching TTS |
| **Phần cứng** | Raspberry Pi 5 (BCM2712, 4x Cortex-A76, 4-8GB RAM) |
| **VAD** | Silero VAD (ONNX, ~2MB) |
| **ASR** | SenseVoiceSmall (sherpa-onnx INT8, ~234M params) |
| **TTS** | Valtec-TTS (~74.8M params, zero-shot cloning) |
| **Background CPU** | ≤ 40% (Audio + VAD) |
| **Active CPU** | ≤ 70% (ASR + TTS) |

## Sơ đồ kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM BOOT (1 lần)                       │
│  Load VAD Model → Load ASR Model → Load TTS Model → Ready   │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │     THREAD 1: Producer (≤40% CPU)    │
        │                                      │
        │  ┌──────────┐    ┌──────────────┐   │
        │  │ Mic PCM  │───▶│ Ring Buffer  │   │
        │  │ 16kHz    │    │ (3s, fixed)  │   │
        │  └──────────┘    └──────┬───────┘   │
        │                         │            │
        │                  ┌──────▼───────┐   │
        │                  │  Silero VAD  │   │
        │                  │  (ONNX, 1T)  │   │
        │                  └──────┬───────┘   │
        │                         │            │
        │              Speech=True?│            │
        └─────────────────────────┼────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  Thread-safe Queue         │
                    │  maxsize=50, Bounded       │
                    │  + Backpressure mechanism  │
                    └─────────────┬─────────────┘
                                  │
        ┌─────────────────────────▼────────────┐
        │     THREAD 2: Consumer (≤70% CPU)     │
        │                                       │
        │  ┌───────────────┐  ┌──────────────┐ │
        │  │ SenseVoice    │  │ Text Norm    │ │
        │  │ ASR (INT8)    │──▶│ Code-switch │ │
        │  │ sherpa-onnx   │  │ Pipeline     │ │
        │  └───────────────┘  └──────┬───────┘ │
        │                            │          │
        │                     ┌──────▼───────┐ │
        │                     │  Valtec-TTS  │ │
        │                     │  (~74.8M)    │ │
        │                     └──────┬───────┘ │
        │                            │          │
        │                     ┌──────▼───────┐ │
        │                     │  Speaker Out │ │
        │                     └──────────────┘ │
        └───────────────────────────────────────┘
```

## Cấu trúc thư mục dự án

```
AI_EDGE_S2/
├── README.md
├── requirements.txt
├── docs/
│   ├── 01_architecture.md        # Kiến trúc tổng thể
│   ├── 02_implementation.md      # Chi tiết code + pseudo-code
│   └── 03_documentation.md       # Giải trình kỹ thuật (Q&A)
├── src/
│   ├── config.py                 # Cấu hình hệ thống
│   ├── ring_buffer.py            # Ring Buffer (fixed-size)
│   ├── vad_engine.py             # Silero VAD wrapper
│   ├── asr_engine.py             # SenseVoiceSmall wrapper
│   ├── tts_engine.py             # Valtec-TTS + Code-switching
│   ├── text_normalizer.py        # Text-norm đa ngôn ngữ
│   ├── pipeline.py               # AlwaysOnPipeline (Producer-Consumer)
│   └── cpu_governor.py           # CPU resource monitor
├── models/                       # Model files (gitignored)
│   ├── silero_vad.onnx
│   ├── sensevoice-small-int8.onnx
│   └── valtec-tts/
├── scripts/
│   ├── setup_pi5.sh
│   └── benchmark.py
└── tests/
    ├── test_ring_buffer.py
    ├── test_vad.py
    └── test_pipeline.py
```

## Lựa chọn model và lý do

### VAD: Silero VAD (ONNX)
- **Kích thước**: ~2MB ONNX
- **CPU**: <5% trên Pi5 với 1 thread ONNX
- **Latency**: <1ms per 32ms chunk
- **Lý do chọn**: Siêu nhẹ, ONNX portable, state-of-the-art accuracy

### ASR: SenseVoiceSmall (sherpa-onnx INT8)
- **Params**: 234M (INT8 quantized → ~60MB)
- **Ngôn ngữ**: Chinese, English, Japanese, Korean, Cantonese
- **Tốc độ**: 15x nhanh hơn Whisper-Large (non-autoregressive)
- **Deploy**: sherpa-onnx framework → native ARM64 support
- **Lưu ý**: Không hỗ trợ native tiếng Việt → cần text post-processing

### TTS: Valtec-TTS
- **Params**: ~74.8M (VITS2 architecture)
- **RTF**: ~0.24 trên CPU → nhanh hơn real-time
- **Feature**: Zero-shot voice cloning từ 3-10s audio
- **Lưu ý**: Chủ yếu optimize cho tiếng Việt → cần Code-switching pipeline
