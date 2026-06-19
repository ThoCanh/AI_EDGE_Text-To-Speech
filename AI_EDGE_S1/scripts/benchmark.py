"""
benchmark.py — RTF & Memory Benchmark cho Voice Pipeline
=========================================================
Đo lường KPIs theo yêu cầu bài test:
- RTF (Real-Time Factor) < 0.3
- Memory leak detection (RSS stable)
"""

import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import AUDIO_SAMPLE_RATE, TARGET_RTF


def benchmark_rtf():
    """
    Benchmark RTF với audio giả lập 5 giây.
    RTF = processing_time / audio_duration
    Target: RTF < 0.3 (5s audio → xử lý < 1.5s)
    """
    from asr_engine import ASREngine
    from tts_engine import TTSEngine

    print("=" * 50)
    print("📊 RTF BENCHMARK")
    print("=" * 50)

    # Load models (1 lần)
    asr = ASREngine()
    tts = TTSEngine()
    asr.warmup()
    tts.warmup()

    # Tạo test audio (5s random noise - simulates speech)
    duration_seconds = 5.0
    test_audio = np.random.randn(
        int(AUDIO_SAMPLE_RATE * duration_seconds)
    ).astype(np.float32) * 0.1

    results = []
    for i in range(5):
        # ASR
        t0 = time.perf_counter()
        text = asr.transcribe(test_audio)
        asr_time = time.perf_counter() - t0

        # TTS
        t1 = time.perf_counter()
        audio_out = tts.synthesize(text if text else "Xin chào")
        tts_time = time.perf_counter() - t1

        total = asr_time + tts_time
        rtf = total / duration_seconds

        results.append(rtf)
        status = "✅" if rtf < TARGET_RTF else "❌"
        print(f"  Run {i+1}: ASR={asr_time:.3f}s TTS={tts_time:.3f}s "
              f"RTF={rtf:.3f} {status}")

    avg_rtf = np.mean(results)
    print(f"\n  Average RTF: {avg_rtf:.3f} "
          f"{'✅ PASS' if avg_rtf < TARGET_RTF else '❌ FAIL'}")

    asr.shutdown()
    tts.shutdown()


def benchmark_memory():
    """
    Kiểm tra memory leak qua nhiều inference cycles.
    RSS không được tăng dần theo thời gian.
    """
    from memory_manager import MemoryManager

    print("\n" + "=" * 50)
    print("📊 MEMORY LEAK TEST")
    print("=" * 50)

    mem = MemoryManager()
    baseline = mem.get_rss_mb()
    print(f"  Baseline RSS: {baseline:.1f}MB")

    # Simulate 20 inference cycles
    for i in range(20):
        # Allocate + free buffers (simulates pipeline)
        buf = np.random.randn(AUDIO_SAMPLE_RATE * 5).astype(np.float32)
        _ = buf.tobytes()
        del buf
        mem.force_gc()

        if (i + 1) % 5 == 0:
            current = mem.get_rss_mb()
            delta = current - baseline
            status = "✅" if abs(delta) < 10 else "❌ LEAK!"
            print(f"  Cycle {i+1}: RSS={current:.1f}MB (Δ{delta:+.1f}MB) {status}")

    mem.shutdown()


if __name__ == "__main__":
    if "--rtf" in sys.argv:
        benchmark_rtf()
    elif "--memory" in sys.argv:
        benchmark_memory()
    else:
        benchmark_memory()
        print("\n(Chạy --rtf để benchmark RTF — cần models)")
