#!/usr/bin/env python3
"""
Entry point - Always-on Voice Assistant Pipeline.

Usage:
    python main.py                  # Chạy pipeline thật
    python main.py --demo           # Demo text normalizer
    python main.py --benchmark      # Benchmark CPU/latency
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-18s] %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(args: argparse.Namespace) -> None:
    """Chạy AlwaysOnPipeline chính."""
    from src.pipeline import AlwaysOnPipeline
    from src.config import VAD, ASR, TTS

    pipeline = AlwaysOnPipeline(
        vad_path=args.vad_model or VAD.model_path,
        asr_path=args.asr_model or ASR.model_path,
        tts_path=args.tts_model or TTS.model_path,
    )

    def _shutdown(sig, frame):
        logger.info("Signal %s received. Shutting down...", sig)
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    pipeline.start()
    try:
        while pipeline.is_running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pipeline.stop()


def run_demo() -> None:
    """Demo Text Normalizer (không cần models)."""
    from src.nlp import CodeSwitchNormalizer, SeverityDetector

    norm = CodeSwitchNormalizer()
    tests = [
        "BMS overcurrent 24V",
        "CAN bus communication timeout",
        "Battery overheat 85V",
        "ECU firmware OTA update",
        "Loi critical inverter shutdown do overcurrent 150A",
    ]

    print("=" * 60)
    print("  CODE-SWITCHING TEXT NORMALIZER DEMO")
    print("=" * 60)
    for t in tests:
        out = norm.normalize(t)
        sev = SeverityDetector.detect(t)
        print(f"\nIN:  {t}")
        print(f"OUT: {out}")
        print(f"SEV: {sev}")

    print("\n" + "=" * 60)
    print("  BENCHMARK: 10000 iterations")
    start = time.perf_counter()
    for _ in range(10000):
        norm.normalize("BMS overcurrent 24V CAN bus timeout")
    elapsed = time.perf_counter() - start
    print(f"  Total: {elapsed:.3f}s | Per call: {elapsed/10000*1000:.4f}ms")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Always-on Voice Assistant Pipeline (AI EDGE S2)",
    )
    parser.add_argument("--demo", action="store_true", help="Demo text normalizer")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark")
    parser.add_argument("--vad-model", type=str, help="VAD ONNX model path")
    parser.add_argument("--asr-model", type=str, help="ASR ONNX model path")
    parser.add_argument("--tts-model", type=str, help="TTS model path")

    args = parser.parse_args()
    if args.demo or args.benchmark:
        run_demo()
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
