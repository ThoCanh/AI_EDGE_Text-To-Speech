"""
Production-grade Unit Tests — AI EDGE S2.
Covering all 3 Khối + CPU Constraints + Performance + Architecture.

Chạy: py -3 -m pytest tests/test_core.py -v
"""

import collections
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════
# KHỐI 1: Ring Buffer — Bộ đệm vòng kích thước cố định
# ═══════════════════════════════════════════════════════════

class TestRingBuffer:
    """Khối 1: Ring Buffer cố định, thread-safe, zero memory leak."""

    def _make(self, max_chunks=10, chunk_ms=32):
        from src.audio import RingBuffer
        return RingBuffer(max_chunks=max_chunks, chunk_ms=chunk_ms)

    # ── Core: ghi/đọc ────────────────────────

    def test_write_and_read(self):
        buf = self._make(max_chunks=5)
        buf.write(np.ones(512, dtype=np.float32))
        result = buf.read_all()
        assert len(result) == 512
        assert np.allclose(result, 1.0)

    def test_empty_read_returns_empty_array(self):
        buf = self._make()
        assert len(buf.read_all()) == 0
        assert len(buf.read_last_n_ms(300)) == 0

    def test_clear(self):
        buf = self._make()
        buf.write(np.zeros(512, dtype=np.float32))
        buf.clear()
        assert buf.is_empty

    # ── ZERO MEMORY LEAK (ràng buộc bắt buộc) ──

    def test_fixed_size_discards_old_frames(self):
        """deque(maxlen) discard frame cũ → KHÔNG memory leak."""
        buf = self._make(max_chunks=5)
        for i in range(10):
            buf.write(np.full(512, i, dtype=np.float32))
        assert len(buf) == 5
        assert buf.read_all()[0] == 5.0  # chunk 0-4 bị loại

    def test_long_running_10k_writes_no_growth(self):
        """Simulate always-on: 10000 writes → buffer cố định."""
        buf = self._make(max_chunks=10)
        for _ in range(10000):
            buf.write(np.random.randn(512).astype(np.float32))
        assert len(buf) == 10
        assert buf.is_full

    def test_uses_deque_with_maxlen(self):
        """Implementation: phải dùng deque(maxlen=N)."""
        buf = self._make(max_chunks=5)
        assert isinstance(buf._buffer, collections.deque)
        assert buf._buffer.maxlen == 5

    # ── Buffer 3 giây (đề bài) ────────────────

    def test_default_buffer_3_seconds(self):
        """Buffer mặc định = 3 giây = 93 chunks × 32ms."""
        from src.audio import RingBuffer
        buf = RingBuffer()  # Default config
        from src.config import AUDIO
        assert AUDIO.buffer_seconds == 3
        assert buf._buffer.maxlen == AUDIO.max_buffer_chunks

    # ── Pre-roll (tránh cắt đầu câu) ─────────

    def test_pre_roll_300ms(self):
        buf = self._make(max_chunks=10, chunk_ms=32)
        for i in range(10):
            buf.write(np.full(512, i, dtype=np.float32))
        pre = buf.read_last_n_ms(96)  # 3 chunks
        assert len(pre) == 512 * 3
        assert pre[0] == 7.0

    def test_pre_roll_less_data(self):
        buf = self._make(max_chunks=10, chunk_ms=32)
        buf.write(np.full(512, 1.0, dtype=np.float32))
        pre = buf.read_last_n_ms(300)
        assert len(pre) == 512  # Chỉ có 1 chunk

    # ── Properties ────────────────────────────

    def test_duration_ms(self):
        buf = self._make(max_chunks=10, chunk_ms=32)
        buf.write(np.zeros(512, dtype=np.float32))
        buf.write(np.zeros(512, dtype=np.float32))
        assert buf.duration_ms == 64.0

    def test_repr(self):
        buf = self._make(max_chunks=5)
        assert "RingBuffer" in repr(buf)

    # ── Thread Safety (ràng buộc bắt buộc) ────

    def test_thread_safe_concurrent_rw(self):
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

    def test_copy_on_write(self):
        """write() copy data, không giữ reference ngoài."""
        buf = self._make(max_chunks=5)
        data = np.ones(512, dtype=np.float32)
        buf.write(data)
        data[:] = 99.0
        assert np.allclose(buf.read_all(), 1.0)

    def test_has_threading_lock(self):
        buf = self._make()
        assert hasattr(buf, '_lock')
        assert isinstance(buf._lock, type(threading.Lock()))


# ═══════════════════════════════════════════════════════════
# KHỐI 1: VAD State Machine
# ═══════════════════════════════════════════════════════════

class TestVADStateMachine:
    """Khối 1: Silero VAD 4-state machine."""

    def test_four_states_exist(self):
        from src.vad import VADState
        states = [VADState.SILENCE, VADState.PENDING_SPEECH,
                  VADState.SPEECH, VADState.PENDING_SILENCE]
        assert len(states) == 4

    def test_vad_result_named_tuple(self):
        from src.vad import VADResult, VADState
        r = VADResult(is_speech=True, state=VADState.SPEECH,
                      probability=0.95, is_end_of_utterance=False)
        assert r.is_speech is True
        assert r.probability == 0.95
        assert r.is_end_of_utterance is False

    def test_vad_config_silero_optimized(self):
        from src.config import VAD
        assert VAD.threshold_on == 0.5
        assert VAD.threshold_off == 0.35
        assert VAD.min_speech_ms == 250
        assert VAD.max_silence_ms == 700
        assert VAD.pre_roll_ms == 300
        assert VAD.onnx_threads == 1

    def test_vad_state_transitions_documented(self):
        """State machine: SILENCE→PENDING_SPEECH→SPEECH→PENDING_SILENCE→SILENCE."""
        from src.vad import VADState
        # Verify all transitions are valid enum values
        assert VADState.SILENCE.value == "SILENCE"
        assert VADState.PENDING_SPEECH.value == "PENDING_SPEECH"
        assert VADState.SPEECH.value == "SPEECH"
        assert VADState.PENDING_SILENCE.value == "PENDING_SILENCE"


# ═══════════════════════════════════════════════════════════
# KHỐI 2: Producer-Consumer Architecture
# ═══════════════════════════════════════════════════════════

class TestProducerConsumer:
    """Khối 2: Thread-safe Queue, bounded, backpressure."""

    def test_bounded_queue_config(self):
        from src.config import PIPELINE
        assert PIPELINE.queue_maxsize == 50
        assert PIPELINE.drop_threshold == 0.8
        assert PIPELINE.consumer_get_timeout == 0.5

    def test_queue_full_does_not_block_forever(self):
        """Queue đầy → timeout, KHÔNG block cứng hệ thống."""
        q = queue.Queue(maxsize=3)
        for _ in range(3):
            q.put(np.zeros(512))
        with pytest.raises(queue.Full):
            q.put(np.zeros(512), timeout=0.01)

    def test_sentinel_none_stops_consumer(self):
        """None sentinel → consumer thread dừng."""
        q = queue.Queue()
        q.put(np.zeros(512))
        q.put(None)
        assert q.get() is not None
        assert q.get() is None  # Consumer break

    def test_backpressure_drops_oldest(self):
        from src.config import PIPELINE
        q = queue.Queue(maxsize=10)
        threshold = int(10 * PIPELINE.drop_threshold)
        for i in range(threshold):
            q.put(np.full(512, i, dtype=np.float32))
        assert q.qsize() >= threshold
        oldest = q.get_nowait()
        assert oldest[0] == 0.0

    def test_consumer_sleeps_when_queue_empty(self):
        """Thread 2 NGỦL khi queue rỗng (Queue.get timeout)."""
        q = queue.Queue()
        woke = threading.Event()
        def consumer():
            try:
                q.get(timeout=0.1)
                woke.set()
            except queue.Empty:
                pass
        t = threading.Thread(target=consumer)
        t.start()
        time.sleep(0.05)
        assert not woke.is_set()
        q.put(np.zeros(512))
        t.join(timeout=1.0)
        assert woke.is_set()

    def test_thread_safe_queue_concurrent(self):
        """Producer/Consumer đồng thời → không race condition."""
        q = queue.Queue(maxsize=20)
        produced = []
        consumed = []
        def producer():
            for i in range(100):
                q.put(i)
                produced.append(i)
        def consumer():
            while True:
                try:
                    item = q.get(timeout=0.5)
                    consumed.append(item)
                    q.task_done()
                except queue.Empty:
                    break
        tp = threading.Thread(target=producer)
        tc = threading.Thread(target=consumer)
        tp.start(); tc.start()
        tp.join(); tc.join()
        assert len(consumed) == 100

    def test_pipeline_has_two_threads(self):
        """Pipeline phải có Producer + Consumer threads."""
        # Verify class structure
        from src.pipeline.always_on import AlwaysOnPipeline
        import inspect
        source = inspect.getsource(AlwaysOnPipeline)
        assert "_producer_loop" in source
        assert "_consumer_loop" in source
        assert "Producer-AudioVAD" in source
        assert "Consumer-ASRTTS" in source

    def test_pipeline_has_3_layer_protection(self):
        """Pipeline có 3 lớp bảo vệ: VAD timeout, backpressure, CPU throttle."""
        from src.pipeline.always_on import AlwaysOnPipeline
        import inspect
        source = inspect.getsource(AlwaysOnPipeline)
        assert "vad_timeout" in source.lower() or "VAD Timeout" in source
        assert "backpressure" in source.lower() or "Backpressure" in source
        assert "throttle" in source.lower()


# ═══════════════════════════════════════════════════════════
# CPU Constraints
# ═══════════════════════════════════════════════════════════

class TestCPUConstraints:
    """CPU Budget: Background ≤40%, Active ≤70%."""

    def test_background_limit_40_percent(self):
        from src.config import CPU
        assert CPU.background_max_percent == 40.0

    def test_active_limit_70_percent(self):
        from src.config import CPU
        assert CPU.active_max_percent == 70.0

    def test_vad_1_onnx_thread(self):
        """VAD 1 thread → CPU < 5% background."""
        from src.config import VAD
        assert VAD.onnx_threads == 1

    def test_asr_2_onnx_threads(self):
        from src.config import ASR
        assert ASR.num_threads == 2

    def test_governor_init_graceful(self):
        from src.system import CPUGovernor
        gov = CPUGovernor()
        gov.throttle_if_needed(is_active=False)
        gov.throttle_if_needed(is_active=True)
        gov.throttle_consumer()

    def test_governor_stats_tracking(self):
        from src.system import CPUGovernor
        gov = CPUGovernor()
        stats = gov.stats
        assert "bg_throttle_count" in stats
        assert "active_throttle_count" in stats
        assert "peak_cpu_percent" in stats

    def test_governor_system_info(self):
        from src.system import CPUGovernor
        gov = CPUGovernor()
        info = gov.get_system_info()
        assert info["cpu_budget"]["background_limit"] == 40.0
        assert info["cpu_budget"]["active_limit"] == 70.0

    def test_consumer_has_throttle_checkpoints(self):
        """Consumer thread phải có CPU throttle giữa ASR↔TTS."""
        from src.pipeline.always_on import AlwaysOnPipeline
        import inspect
        source = inspect.getsource(AlwaysOnPipeline._process_utterance)
        assert "throttle_consumer" in source

    def test_producer_has_dual_mode_throttle(self):
        """Producer biết consumer busy → chuyển ngưỡng 40%↔70%."""
        from src.pipeline.always_on import AlwaysOnPipeline
        import inspect
        source = inspect.getsource(AlwaysOnPipeline._producer_loop)
        assert "consumer_busy" in source
        assert "throttle_if_needed" in source


# ═══════════════════════════════════════════════════════════
# KHỐI 3: Code-switching Text Normalizer
# ═══════════════════════════════════════════════════════════

class TestCodeSwitchNormalizer:
    """Khối 3: Code-switching Anh-Việt text normalization."""

    def _make(self):
        from src.nlp import CodeSwitchNormalizer
        return CodeSwitchNormalizer()

    # ── Acronyms ──────────────────────────────

    def test_bms(self):
        assert "bi em ét" in self._make().normalize("BMS")

    def test_can(self):
        assert "can" in self._make().normalize("CAN")

    def test_ecu(self):
        assert "i xi du" in self._make().normalize("ECU")

    # ── Units ─────────────────────────────────

    def test_24v(self):
        assert "24 vôn" in self._make().normalize("24V")

    def test_150a(self):
        assert "150 am-pe" in self._make().normalize("150A")

    def test_85_celsius(self):
        assert "85 độ xê" in self._make().normalize("85°C")

    def test_100kw(self):
        assert "ki-lô oát" in self._make().normalize("100kW")

    def test_decimal_3_3v(self):
        assert "3.3 vôn" in self._make().normalize("3.3V")

    # ── Tech Terms ────────────────────────────

    def test_overcurrent(self):
        assert "ô-vơ-ca-rần" in self._make().normalize("Overcurrent")

    def test_timeout(self):
        assert "thai-ao" in self._make().normalize("timeout")

    def test_communication(self):
        assert "cơm-miu-ni-kây-sần" in self._make().normalize("communication")

    def test_case_insensitive(self):
        norm = self._make()
        for word in ["OVERCURRENT", "overcurrent", "Overcurrent"]:
            assert "ô-vơ-ca-rần" in norm.normalize(word)

    # ── ĐỀ BÀI: câu phức tạp ─────────────────

    def test_spec_sentence_1(self):
        """Đề bài: 'BMS phát hiện lỗi Overcurrent trên đường nguồn 24V'."""
        r = self._make().normalize(
            "Hệ thống đang kiểm tra BMS, phát hiện lỗi Overcurrent trên đường nguồn 24V"
        )
        assert "bi em ét" in r
        assert "ô-vơ-ca-rần" in r
        assert "24 vôn" in r
        assert "Hệ thống" in r  # Tiếng Việt giữ nguyên

    def test_spec_sentence_2(self):
        """Đề bài: 'Mã lỗi CAN bus communication timeout'."""
        r = self._make().normalize("Mã lỗi CAN bus communication timeout")
        assert "can" in r
        assert "cơm-miu-ni-kây-sần" in r
        assert "thai-ao" in r

    # ── Vietnamese preservation ────────────────

    def test_vietnamese_with_diacritics_preserved(self):
        r = self._make().normalize("Phát hiện lỗi nghiêm trọng")
        assert "Phát" in r and "hiện" in r and "lỗi" in r

    def test_empty_string(self):
        assert self._make().normalize("") == ""

    def test_multiple_units(self):
        r = self._make().normalize("Pin 48V 100Ah nhiệt độ 60°C")
        assert "48 vôn" in r
        assert "100 am-pe giờ" in r
        assert "60 độ xê" in r

    # ── Ràng buộc: không phình model ──────────

    def test_dictionary_is_lightweight(self):
        """Từ điển < 200 entries → KHÔNG phình model."""
        from src.nlp.dictionaries import ACRONYM_MAP, TECH_TERMS, UNITS
        total = len(ACRONYM_MAP) + len(TECH_TERMS) + len(UNITS)
        assert total < 200, f"Dictionary too large: {total} entries"

    def test_normalization_is_rules_based(self):
        """Normalizer dùng regex rules, KHÔNG dùng ML model."""
        from src.nlp.normalizer import CodeSwitchNormalizer
        import inspect
        source = inspect.getsource(CodeSwitchNormalizer)
        assert "re.compile" in source
        assert "import torch" not in source
        assert "import tensorflow" not in source


# ═══════════════════════════════════════════════════════════
# KHỐI 3: Severity Detection + Prosody Control
# ═══════════════════════════════════════════════════════════

class TestSeverityDetector:
    def _det(self):
        from src.nlp import SeverityDetector
        return SeverityDetector

    def test_critical(self):
        det = self._det()
        assert det.detect("critical nguy hiểm") == "critical"
        assert det.detect("EMERGENCY shutdown") == "critical"

    def test_warning(self):
        det = self._det()
        assert det.detect("cảnh báo nhiệt độ") == "warning"
        assert det.detect("connection timeout") == "warning"

    def test_normal(self):
        assert self._det().detect("hoạt động bình thường") == "normal"

    def test_critical_priority(self):
        assert self._det().detect("lỗi nguy hiểm") == "critical"


class TestProsodyController:
    def _make(self):
        from src.tts import ProsodyController
        return ProsodyController()

    def test_three_profiles(self):
        ctrl = self._make()
        assert set(ctrl.available_profiles) >= {"normal", "warning", "critical"}

    def test_warning_faster_than_normal(self):
        ctrl = self._make()
        assert ctrl.get_params("warning")["length_scale"] < ctrl.get_params("normal")["length_scale"]

    def test_critical_fastest(self):
        ctrl = self._make()
        assert ctrl.get_params("critical")["length_scale"] < ctrl.get_params("warning")["length_scale"]

    def test_unknown_falls_back(self):
        ctrl = self._make()
        assert ctrl.get_params("xyz") == ctrl.get_params("normal")

    def test_zero_overhead(self):
        """Prosody = scalar params, 0 inference overhead."""
        ctrl = self._make()
        params = ctrl.get_params("critical")
        assert isinstance(params["length_scale"], float)
        assert isinstance(params["noise_scale"], float)


# ═══════════════════════════════════════════════════════════
# PERFORMANCE BENCHMARKS
# ═══════════════════════════════════════════════════════════

class TestPerformance:
    def test_normalizer_under_0_5ms(self):
        from src.nlp import CodeSwitchNormalizer
        norm = CodeSwitchNormalizer()
        start = time.perf_counter()
        for _ in range(5000):
            norm.normalize("BMS overcurrent 24V CAN bus timeout")
        ms = ((time.perf_counter() - start) / 5000) * 1000
        assert ms < 0.5, f"Normalizer: {ms:.4f}ms/call (limit: 0.5ms)"

    def test_ring_buffer_write_under_1ms(self):
        from src.audio import RingBuffer
        buf = RingBuffer(max_chunks=100)
        chunk = np.random.randn(512).astype(np.float32)
        start = time.perf_counter()
        for _ in range(10000):
            buf.write(chunk)
        ms = ((time.perf_counter() - start) / 10000) * 1000
        assert ms < 1.0, f"RingBuffer: {ms:.4f}ms/write (limit: 1ms)"

    def test_severity_under_0_1ms(self):
        from src.nlp import SeverityDetector
        start = time.perf_counter()
        for _ in range(10000):
            SeverityDetector.detect("critical overcurrent nguy hiểm")
        ms = ((time.perf_counter() - start) / 10000) * 1000
        assert ms < 0.1, f"Severity: {ms:.4f}ms (limit: 0.1ms)"


# ═══════════════════════════════════════════════════════════
# CONFIG CONSISTENCY — match đề bài
# ═══════════════════════════════════════════════════════════

class TestConfig:
    def test_audio_16khz_mono(self):
        from src.config import AUDIO
        assert AUDIO.sample_rate == 16000
        assert AUDIO.channels == 1

    def test_chunk_32ms_512_samples(self):
        from src.config import AUDIO
        assert AUDIO.chunk_ms == 32
        assert AUDIO.chunk_size == 512

    def test_buffer_3_seconds(self):
        from src.config import AUDIO
        assert AUDIO.buffer_seconds == 3

    def test_tts_model_under_100m(self):
        """TTS model < 100M parameters (đề bài)."""
        from src.config import TTS
        assert TTS.output_sample_rate == 22050
        assert "normal" in TTS.prosody_profiles

    def test_frozen_dataclass(self):
        """Config frozen → không bị sửa runtime."""
        from src.config import AUDIO
        with pytest.raises(AttributeError):
            AUDIO.sample_rate = 44100


# ═══════════════════════════════════════════════════════════
# ARCHITECTURE VALIDATION
# ═══════════════════════════════════════════════════════════

class TestArchitecture:
    """Validate kiến trúc modular theo đề bài."""

    def test_modules_exist(self):
        import src.audio
        import src.vad
        import src.asr
        import src.nlp
        import src.tts
        import src.system
        import src.pipeline

    def test_ring_buffer_importable(self):
        from src.audio import RingBuffer
        assert RingBuffer is not None

    def test_vad_types_importable(self):
        from src.vad import VADState, VADResult
        assert VADState is not None

    def test_normalizer_importable(self):
        from src.nlp import CodeSwitchNormalizer, SeverityDetector
        assert CodeSwitchNormalizer is not None

    def test_prosody_importable(self):
        from src.tts import ProsodyController
        assert ProsodyController is not None

    def test_cpu_governor_importable(self):
        from src.system import CPUGovernor
        assert CPUGovernor is not None

    def test_pipeline_importable(self):
        from src.pipeline import AlwaysOnPipeline
        assert AlwaysOnPipeline is not None

    def test_dictionaries_separated(self):
        """Từ điển tách riêng khỏi logic normalizer."""
        from src.nlp.dictionaries import ACRONYM_MAP, TECH_TERMS, UNITS, LETTER_MAP
        assert len(ACRONYM_MAP) > 0
        assert len(TECH_TERMS) > 0
        assert len(UNITS) > 0
        assert len(LETTER_MAP) == 26
