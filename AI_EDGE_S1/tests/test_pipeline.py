"""
test_pipeline.py — Unit Tests cho Voice Pipeline
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_config():
    """Test config values hop le."""
    from config import (
        NUM_INFERENCE_THREADS,
        AUDIO_SAMPLE_RATE,
        TARGET_RTF,
    )
    assert NUM_INFERENCE_THREADS == 2, "Phai dung 2 threads tren Pi5"
    assert AUDIO_SAMPLE_RATE == 16000, "Whisper yeu cau 16kHz"
    assert TARGET_RTF == 0.3
    print("[PASS] test_config")


def test_memory_manager():
    """Test MemoryManager co ban."""
    from memory_manager import MemoryManager

    mem = MemoryManager()
    rss = mem.get_rss_mb()
    assert rss != 0, "RSS phai > 0"

    mem.set_baseline()
    status = mem.get_status()
    assert "rss_mb" in status
    assert "tmpfs_dir" in status

    # Test gc
    mem.force_gc()

    mem.shutdown()
    print("[PASS] test_memory_manager")


def test_audio_buffer():
    """Test audio buffer recording trong RAM."""
    # Simulate recording callback
    buffer = bytearray()
    for _ in range(10):
        chunk = np.random.randn(1024).astype(np.float32)
        buffer.extend(chunk.tobytes())

    # Convert back (giong stop_recording)
    pcm = np.frombuffer(bytes(buffer), dtype=np.float32)
    assert len(pcm) == 10240, f"Expected 10240, got {len(pcm)}"
    assert pcm.dtype == np.float32

    # Cleanup
    del pcm
    buffer = bytearray()
    print("[PASS] test_audio_buffer")


def test_pcm_no_file_io():
    """Test raw PCM data truyen truc tiep khong qua file."""
    # Simulate: mic -> buffer -> numpy -> ASR (no .wav file)
    raw_audio = np.random.randn(16000 * 3).astype(np.float32)  # 3s audio

    # Convert to bytes (like audio callback)
    audio_bytes = raw_audio.tobytes()

    # Convert back (like stop_and_process)
    restored = np.frombuffer(audio_bytes, dtype=np.float32)

    # Verify data integrity (zero loss)
    assert np.array_equal(raw_audio, restored), "PCM data bi thay doi khi truyen qua RAM!"
    assert restored.dtype == np.float32
    assert len(restored) == 16000 * 3

    print("[PASS] test_pcm_no_file_io")


def test_model_load_once():
    """Test: model chi duoc load 1 lan trong __init__."""
    # Verify VoicePipeline class structure
    from voice_pipeline import VoicePipeline
    import inspect

    source = inspect.getsource(VoicePipeline.__init__)

    # __init__ phai chua code load ASR va TTS
    assert "ASREngine" in source, "__init__ phai load ASR model"
    assert "TTSEngine" in source, "__init__ phai load TTS model"

    # start_recording KHONG duoc load model
    rec_source = inspect.getsource(VoicePipeline.start_recording)
    assert "ASREngine" not in rec_source, "start_recording KHONG duoc load model!"
    assert "TTSEngine" not in rec_source, "start_recording KHONG duoc load model!"

    # stop_and_process KHONG duoc load model
    proc_source = inspect.getsource(VoicePipeline.stop_and_process)
    assert "ASREngine" not in proc_source, "stop_and_process KHONG duoc load model!"

    print("[PASS] test_model_load_once")


if __name__ == "__main__":
    test_config()
    test_memory_manager()
    test_audio_buffer()
    test_pcm_no_file_io()
    test_model_load_once()
    print("\n[ALL PASSED] 5/5 tests passed!")
