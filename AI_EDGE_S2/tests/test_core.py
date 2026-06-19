"""
Unit tests cho Ring Buffer, Text Normalizer, và Severity Detector.
Chạy: python -m pytest tests/ -v
"""

import threading

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════
# Ring Buffer Tests (Khối 1)
# ═══════════════════════════════════════════════════════════

class TestRingBuffer:
    """Test: Ring Buffer kích thước cố định, thread-safe."""

    def _make(self, max_chunks=10, chunk_ms=32):
        from src.audio import RingBuffer
        return RingBuffer(max_chunks=max_chunks, chunk_ms=chunk_ms)

    def test_write_and_read(self):
        buf = self._make(max_chunks=5)
        buf.write(np.ones(512, dtype=np.float32))
        result = buf.read_all()
        assert len(result) == 512
        assert np.allclose(result, 1.0)

    def test_fixed_size_no_memory_leak(self):
        """deque(maxlen) discard frame cũ → KHÔNG memory leak."""
        buf = self._make(max_chunks=5)
        for i in range(10):
            buf.write(np.full(512, i, dtype=np.float32))
        assert len(buf) == 5
        assert buf.read_all()[0] == 5.0  # chunk 0-4 bị loại

    def test_pre_roll(self):
        buf = self._make(max_chunks=10, chunk_ms=32)
        for i in range(10):
            buf.write(np.full(512, i, dtype=np.float32))
        pre = buf.read_last_n_ms(96)  # 3 chunks
        assert len(pre) == 512 * 3
        assert pre[0] == 7.0

    def test_clear(self):
        buf = self._make()
        buf.write(np.zeros(512, dtype=np.float32))
        buf.clear()
        assert buf.is_empty

    def test_thread_safety(self):
        buf = self._make(max_chunks=100)
        errors = []

        def writer():
            try:
                for _ in range(500):
                    buf.write(np.random.randn(512).astype(np.float32))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(500):
                    buf.read_all()
                    buf.read_last_n_ms(100)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start(); t2.start()
        t1.join(); t2.join()
        assert not errors

    def test_empty_read(self):
        buf = self._make()
        assert len(buf.read_all()) == 0
        assert len(buf.read_last_n_ms(300)) == 0


# ═══════════════════════════════════════════════════════════
# Text Normalizer Tests (Khối 3)
# ═══════════════════════════════════════════════════════════

class TestCodeSwitchNormalizer:
    """Test: Code-switching Anh-Việt."""

    def _make(self):
        from src.nlp import CodeSwitchNormalizer
        return CodeSwitchNormalizer()

    def test_acronym(self):
        assert "bi em ét" in self._make().normalize("BMS")

    def test_unit(self):
        assert "24 vôn" in self._make().normalize("24V")

    def test_tech_term(self):
        assert "ô-vơ-ca-rần" in self._make().normalize("Overcurrent")

    def test_complex_sentence(self):
        r = self._make().normalize(
            "BMS Overcurrent 24V"
        )
        assert "bi em ét" in r
        assert "ô-vơ-ca-rần" in r
        assert "24 vôn" in r

    def test_can_bus(self):
        r = self._make().normalize("CAN bus communication timeout")
        assert "can" in r
        assert "thai-ao" in r

    def test_temperature(self):
        assert "85 độ xê" in self._make().normalize("85°C")

    def test_empty(self):
        assert self._make().normalize("") == ""


class TestSeverityDetector:
    """Test: Phát hiện severity level."""

    def _det(self):
        from src.nlp import SeverityDetector
        return SeverityDetector

    def test_critical(self):
        assert self._det().detect("critical nguy hiểm") == "critical"

    def test_warning(self):
        assert self._det().detect("cảnh báo nhiệt độ") == "warning"

    def test_normal(self):
        assert self._det().detect("hoạt động bình thường") == "normal"
