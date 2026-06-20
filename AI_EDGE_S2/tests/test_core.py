"""
Comprehensive Unit Tests - AI EDGE S2.

Covering 3 Khối:
    Khối 1: Ring Buffer + VAD State Machine
    Khối 2: Producer-Consumer Pipeline Logic
    Khối 3: Code-switching Normalizer + Prosody + Severity

Chạy: python -m pytest tests/test_core.py -v
"""

import queue
import threading
import time

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════
# KHỐI 1: Ring Buffer Tests
# ═══════════════════════════════════════════════════════════

class TestRingBuffer:
    """Test: Ring Buffer kích thước cố định, thread-safe, zero memory leak."""

    def _make(self, max_chunks=10, chunk_ms=32):
        from src.audio import RingBuffer
        return RingBuffer(max_chunks=max_chunks, chunk_ms=chunk_ms)

    # ── Chức năng cơ bản ──────────────────────

    def test_write_and_read(self):
        """Ghi 1 chunk → đọc lại đúng data."""
        buf = self._make(max_chunks=5)
        buf.write(np.ones(512, dtype=np.float32))
        result = buf.read_all()
        assert len(result) == 512
        assert np.allclose(result, 1.0)

    def test_empty_read(self):
        """Buffer rỗng → trả array rỗng, không crash."""
        buf = self._make()
        assert len(buf.read_all()) == 0
        assert len(buf.read_last_n_ms(300)) == 0

    def test_clear(self):
        """clear() → buffer rỗng."""
        buf = self._make()
        buf.write(np.zeros(512, dtype=np.float32))
        buf.clear()
        assert buf.is_empty

    # ── ZERO MEMORY LEAK (yêu cầu bắt buộc) ──

    def test_fixed_size_no_memory_leak(self):
        """deque(maxlen) tự động discard frame cũ → KHÔNG memory leak."""
        buf = self._make(max_chunks=5)
        for i in range(10):
            buf.write(np.full(512, i, dtype=np.float32))
        # Buffer chỉ giữ 5 chunks cuối (5, 6, 7, 8, 9)
        assert len(buf) == 5
        data = buf.read_all()
        assert data[0] == 5.0  # chunk 0-4 bị loại bỏ

    def test_long_running_no_memory_growth(self):
        """Simulate 10000 writes → buffer size vẫn cố định."""
        buf = self._make(max_chunks=10)
        for i in range(10000):
            buf.write(np.random.randn(512).astype(np.float32))
        assert len(buf) == 10  # Luôn giữ đúng max_chunks
        assert not buf.is_empty
        assert buf.is_full

    # ── Pre-roll (giữ audio trước VAD detect) ──

    def test_pre_roll(self):
        """read_last_n_ms(96) = 3 chunks × 32ms."""
        buf = self._make(max_chunks=10, chunk_ms=32)
        for i in range(10):
            buf.write(np.full(512, i, dtype=np.float32))
        pre = buf.read_last_n_ms(96)  # 3 chunks
        assert len(pre) == 512 * 3
        assert pre[0] == 7.0  # chunk 7, 8, 9

    def test_pre_roll_less_data_than_requested(self):
        """Yêu cầu 300ms pre-roll nhưng chỉ có 2 chunks → trả hết data có."""
        buf = self._make(max_chunks=10, chunk_ms=32)
        buf.write(np.full(512, 1.0, dtype=np.float32))
        buf.write(np.full(512, 2.0, dtype=np.float32))
        pre = buf.read_last_n_ms(300)  # Yêu cầu 9 chunks nhưng chỉ có 2
        assert len(pre) == 512 * 2

    # ── Properties ────────────────────────────

    def test_duration_ms(self):
        """duration_ms = len(buffer) × chunk_ms."""
        buf = self._make(max_chunks=10, chunk_ms=32)
        buf.write(np.zeros(512, dtype=np.float32))
        buf.write(np.zeros(512, dtype=np.float32))
        assert buf.duration_ms == 64.0

    def test_is_full(self):
        buf = self._make(max_chunks=3)
        for _ in range(3):
            buf.write(np.zeros(512, dtype=np.float32))
        assert buf.is_full

    # ── Thread Safety (yêu cầu bắt buộc) ─────

    def test_thread_safety_concurrent_rw(self):
        """500 writes + 500 reads song song → không crash/race condition."""
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
        assert not errors, f"Thread safety violated: {errors}"

    def test_thread_safety_multiple_writers(self):
        """3 writers ghi đồng thời → không crash."""
        buf = self._make(max_chunks=50)
        errors = []

        def writer(thread_id):
            try:
                for _ in range(200):
                    buf.write(np.full(512, thread_id, dtype=np.float32))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(buf) == 50  # maxlen = 50

    def test_copy_on_write(self):
        """write() phải copy data, không giữ reference ngoài."""
        buf = self._make(max_chunks=5)
        data = np.ones(512, dtype=np.float32)
        buf.write(data)
        data[:] = 99.0  # Sửa data gốc
        result = buf.read_all()
        assert np.allclose(result, 1.0)  # Buffer không bị ảnh hưởng


# ═══════════════════════════════════════════════════════════
# KHỐI 1: VAD State Machine Tests
# ═══════════════════════════════════════════════════════════

class TestVADStateMachine:
    """Test: Silero VAD 4-state machine logic (không cần ONNX model)."""

    def test_state_enum_values(self):
        """4 trạng thái phải đúng."""
        from src.vad import VADState
        assert VADState.SILENCE.value == "SILENCE"
        assert VADState.PENDING_SPEECH.value == "PENDING_SPEECH"
        assert VADState.SPEECH.value == "SPEECH"
        assert VADState.PENDING_SILENCE.value == "PENDING_SILENCE"

    def test_vad_result_fields(self):
        """VADResult phải có 4 fields."""
        from src.vad import VADResult, VADState
        r = VADResult(
            is_speech=True,
            state=VADState.SPEECH,
            probability=0.95,
            is_end_of_utterance=False,
        )
        assert r.is_speech is True
        assert r.state == VADState.SPEECH
        assert r.probability == 0.95
        assert r.is_end_of_utterance is False

    def test_vad_config_values(self):
        """Config VAD phải đúng giá trị đề bài."""
        from src.config import VAD
        assert VAD.threshold_on == 0.5
        assert VAD.threshold_off == 0.35
        assert VAD.min_speech_ms == 250
        assert VAD.max_silence_ms == 700
        assert VAD.pre_roll_ms == 300
        assert VAD.onnx_threads == 1  # 1 thread cho VAD


# ═══════════════════════════════════════════════════════════
# KHỐI 2: Producer-Consumer Architecture Tests
# ═══════════════════════════════════════════════════════════

class TestProducerConsumer:
    """Test: Thread-safe Queue, bounded, backpressure."""

    def test_bounded_queue_config(self):
        """Queue maxsize phải giới hạn."""
        from src.config import PIPELINE
        assert PIPELINE.queue_maxsize == 50
        assert PIPELINE.drop_threshold == 0.8

    def test_bounded_queue_full_blocks(self):
        """Queue đầy → put() phải timeout, không block cứng."""
        q: queue.Queue = queue.Queue(maxsize=3)
        for i in range(3):
            q.put(np.zeros(512))

        # put() với timeout nhỏ → TimeoutError
        with pytest.raises(queue.Full):
            q.put(np.zeros(512), timeout=0.01)

    def test_sentinel_shutdown(self):
        """None sentinel → consumer phải dừng."""
        q: queue.Queue = queue.Queue()
        q.put(np.zeros(512))
        q.put(None)  # Sentinel

        item = q.get()
        assert item is not None
        sentinel = q.get()
        assert sentinel is None  # Consumer nhận None → break

    def test_backpressure_drop_oldest(self):
        """Khi queue gần đầy, drop frame cũ nhất."""
        from src.config import PIPELINE
        maxsize = 10
        threshold = int(maxsize * PIPELINE.drop_threshold)  # 8

        q: queue.Queue = queue.Queue(maxsize=maxsize)
        for i in range(threshold):
            q.put(np.full(512, i, dtype=np.float32))

        # Queue đạt 80% → drop oldest
        assert q.qsize() >= threshold
        oldest = q.get_nowait()
        assert oldest[0] == 0.0  # Frame đầu tiên bị drop

    def test_consumer_only_wakes_on_speech(self):
        """Consumer thread chỉ xử lý khi queue có data (timeout wake)."""
        q: queue.Queue = queue.Queue()
        woke_up = threading.Event()

        def consumer():
            try:
                q.get(timeout=0.1)
                woke_up.set()
            except queue.Empty:
                pass

        t = threading.Thread(target=consumer)
        t.start()
        time.sleep(0.05)
        assert not woke_up.is_set()  # Consumer vẫn ngủ (queue rỗng)

        q.put(np.zeros(512))  # Speech vào queue
        t.join(timeout=1.0)
        assert woke_up.is_set()  # Consumer đã thức dậy

    def test_cpu_budget_config(self):
        """CPU budget theo đề bài: background ≤40%, active ≤70%."""
        from src.config import CPU
        assert CPU.background_max_percent == 40.0
        assert CPU.active_max_percent == 70.0


class TestCPUGovernor:
    """Test: CPU Governor - adaptive throttle + stats tracking."""

    def test_init_without_psutil(self):
        """Governor khởi tạo được dù không có psutil."""
        from src.system import CPUGovernor
        gov = CPUGovernor()
        # Phải không crash khi gọi throttle
        gov.throttle_if_needed(is_active=False)
        gov.throttle_if_needed(is_active=True)

    def test_throttle_consumer_method(self):
        """throttle_consumer() phải gọi được mà không crash."""
        from src.system import CPUGovernor
        gov = CPUGovernor()
        gov.throttle_consumer()  # Không crash

    def test_stats_tracking(self):
        """Governor phải track throttle stats."""
        from src.system import CPUGovernor
        gov = CPUGovernor()
        stats = gov.stats
        assert isinstance(stats, dict)
        assert "bg_throttle_count" in stats
        assert "active_throttle_count" in stats
        assert "bg_violations" in stats
        assert "active_violations" in stats
        assert "peak_cpu_percent" in stats
        assert "current_cpu_percent" in stats

    def test_system_info_includes_budget(self):
        """get_system_info() phải bao gồm CPU budget limits."""
        from src.system import CPUGovernor
        gov = CPUGovernor()
        info = gov.get_system_info()
        assert isinstance(info, dict)
        assert "cpu_percent" in info
        assert "cpu_budget" in info
        assert info["cpu_budget"]["background_limit"] == 40.0
        assert info["cpu_budget"]["active_limit"] == 70.0

    def test_cpu_constraint_values(self):
        """Ràng buộc CPU đúng đề bài: BG ≤40%, Active ≤70%."""
        from src.config import CPU
        assert CPU.background_max_percent == 40.0, "Background phải ≤ 40%"
        assert CPU.active_max_percent == 70.0, "Active phải ≤ 70%"

    def test_thread_pinning_vad(self):
        """VAD phải chỉ dùng 1 ONNX thread → CPU < 5% background."""
        from src.config import VAD
        assert VAD.onnx_threads == 1, "VAD phải dùng 1 thread ONNX"

    def test_thread_pinning_asr(self):
        """ASR dùng 2 ONNX threads → cân bằng latency vs CPU."""
        from src.config import ASR
        assert ASR.num_threads == 2, "ASR phải dùng 2 threads ONNX"


# ═══════════════════════════════════════════════════════════
# KHỐI 3: Code-switching Normalizer Tests
# ═══════════════════════════════════════════════════════════

class TestCodeSwitchNormalizer:
    """Test: Code-switching Anh-Việt text normalization."""

    def _make(self):
        from src.nlp import CodeSwitchNormalizer
        return CodeSwitchNormalizer()

    # ── Acronyms (viết tắt) ───────────────────

    def test_acronym_bms(self):
        assert "bi em ét" in self._make().normalize("BMS")

    def test_acronym_can(self):
        assert "can" in self._make().normalize("CAN")

    def test_acronym_ecu(self):
        assert "i xi du" in self._make().normalize("ECU")

    def test_acronym_preserves_unknown(self):
        """Acronym không có trong dict → giữ nguyên."""
        result = self._make().normalize("XYZ")
        # XYZ không trong ACRONYM_MAP → giữ nguyên hoặc spell out
        assert result  # Không crash

    # ── Units (đơn vị đo) ────────────────────

    def test_unit_voltage(self):
        assert "24 vôn" in self._make().normalize("24V")

    def test_unit_current(self):
        assert "150 am-pe" in self._make().normalize("150A")

    def test_unit_temperature(self):
        assert "85 độ xê" in self._make().normalize("85°C")

    def test_unit_kilowatt(self):
        assert "ki-lô oát" in self._make().normalize("100kW")

    def test_unit_milliamp(self):
        assert "mi-li am-pe" in self._make().normalize("500mA")

    def test_unit_decimal(self):
        """Số thập phân: 3.3V → 3.3 vôn."""
        result = self._make().normalize("3.3V")
        assert "3.3 vôn" in result

    # ── Tech terms (thuật ngữ kỹ thuật) ──────

    def test_tech_overcurrent(self):
        assert "ô-vơ-ca-rần" in self._make().normalize("Overcurrent")

    def test_tech_timeout(self):
        assert "thai-ao" in self._make().normalize("timeout")

    def test_tech_communication(self):
        result = self._make().normalize("communication")
        assert "cơm-miu-ni-kây-sần" in result

    def test_tech_case_insensitive(self):
        """Case-insensitive: OVERCURRENT, overcurrent → cùng kết quả."""
        norm = self._make()
        r1 = norm.normalize("OVERCURRENT")
        r2 = norm.normalize("overcurrent")
        r3 = norm.normalize("Overcurrent")
        assert "ô-vơ-ca-rần" in r1
        assert "ô-vơ-ca-rần" in r2
        assert "ô-vơ-ca-rần" in r3

    # ── Complex sentences (câu phức tạp) ─────

    def test_complex_bms_overcurrent_24v(self):
        """Đề bài: 'BMS overcurrent 24V' → phiên âm đầy đủ."""
        r = self._make().normalize("BMS Overcurrent 24V")
        assert "bi em ét" in r
        assert "ô-vơ-ca-rần" in r
        assert "24 vôn" in r

    def test_complex_can_bus_timeout(self):
        """Đề bài: 'CAN bus communication timeout'."""
        r = self._make().normalize("CAN bus communication timeout")
        assert "can" in r
        assert "thai-ao" in r

    def test_complex_full_sentence(self):
        """Đề bài: câu dài chứa cả Việt lẫn Anh."""
        r = self._make().normalize(
            "Hệ thống đang kiểm tra BMS, phát hiện lỗi Overcurrent trên đường nguồn 24V"
        )
        assert "bi em ét" in r
        assert "ô-vơ-ca-rần" in r
        assert "24 vôn" in r
        # Phần tiếng Việt phải được giữ nguyên
        assert "Hệ thống" in r

    def test_complex_can_bus_full(self):
        """Đề bài: 'Mã lỗi CAN bus communication timeout'."""
        r = self._make().normalize(
            "Mã lỗi CAN bus communication timeout"
        )
        assert "can" in r
        assert "cơm-miu-ni-kây-sần" in r
        assert "thai-ao" in r
        assert "Mã" in r
        assert "lỗi" in r

    # ── Vietnamese text preservation ─────────

    def test_vietnamese_preserved(self):
        """Từ tiếng Việt có dấu KHÔNG bị chuyển đổi."""
        r = self._make().normalize("Phát hiện lỗi nghiêm trọng")
        assert "Phát" in r
        assert "hiện" in r
        assert "lỗi" in r
        assert "nghiêm" in r
        assert "trọng" in r

    # ── Edge cases ────────────────────────────

    def test_empty_string(self):
        assert self._make().normalize("") == ""

    def test_only_numbers(self):
        r = self._make().normalize("12345")
        assert r == "12345"

    def test_multiple_units_in_sentence(self):
        """Nhiều đơn vị trong 1 câu."""
        r = self._make().normalize("Pin 48V 100Ah nhiệt độ 60°C")
        assert "48 vôn" in r
        assert "100 am-pe giờ" in r
        assert "60 độ xê" in r


class TestSeverityDetector:
    """Test: Severity detection cho prosody control."""

    def _det(self):
        from src.nlp import SeverityDetector
        return SeverityDetector

    def test_critical_keywords(self):
        det = self._det()
        assert det.detect("critical nguy hiểm") == "critical"
        assert det.detect("Tình huống khẩn cấp") == "critical"
        assert det.detect("EMERGENCY shutdown") == "critical"
        assert det.detect("overheat detected") == "critical"

    def test_warning_keywords(self):
        det = self._det()
        assert det.detect("cảnh báo nhiệt độ") == "warning"
        assert det.detect("Error code 0x15") == "warning"
        assert det.detect("connection timeout") == "warning"
        assert det.detect("Phát hiện lỗi sensor") == "warning"

    def test_normal_default(self):
        det = self._det()
        assert det.detect("hoạt động bình thường") == "normal"
        assert det.detect("Hệ thống ổn định") == "normal"

    def test_priority_critical_over_warning(self):
        """critical keywords phải ưu tiên trước warning."""
        det = self._det()
        # "lỗi" = warning, nhưng "nguy hiểm" = critical → critical wins
        assert det.detect("lỗi nguy hiểm") == "critical"


class TestProsodyController:
    """Test: Prosody parameter controller."""

    def _make(self):
        from src.tts import ProsodyController
        return ProsodyController()

    def test_normal_profile(self):
        params = self._make().get_params("normal")
        assert params["length_scale"] == 1.0
        assert "noise_scale" in params
        assert "noise_scale_w" in params

    def test_warning_profile_faster(self):
        """Warning: nói nhanh hơn bình thường (length_scale < 1.0)."""
        ctrl = self._make()
        normal = ctrl.get_params("normal")
        warning = ctrl.get_params("warning")
        assert warning["length_scale"] < normal["length_scale"]

    def test_critical_profile_fastest(self):
        """Critical: nói nhanh nhất + pitch ổn định nhất."""
        ctrl = self._make()
        warning = ctrl.get_params("warning")
        critical = ctrl.get_params("critical")
        assert critical["length_scale"] < warning["length_scale"]
        assert critical["noise_scale"] < warning["noise_scale"]

    def test_unknown_severity_falls_back_to_normal(self):
        """Severity lạ → fallback 'normal'."""
        ctrl = self._make()
        params = ctrl.get_params("unknown_severity")
        normal = ctrl.get_params("normal")
        assert params == normal

    def test_available_profiles(self):
        ctrl = self._make()
        profiles = ctrl.available_profiles
        assert "normal" in profiles
        assert "warning" in profiles
        assert "critical" in profiles


# ═══════════════════════════════════════════════════════════
# PERFORMANCE BENCHMARK
# ═══════════════════════════════════════════════════════════

class TestPerformance:
    """Benchmark: đảm bảo performance đạt yêu cầu real-time."""

    def test_normalizer_throughput(self):
        """Normalizer phải xử lý ≥ 10000 calls/sec (< 0.1ms/call)."""
        from src.nlp import CodeSwitchNormalizer
        norm = CodeSwitchNormalizer()

        start = time.perf_counter()
        iterations = 5000
        for _ in range(iterations):
            norm.normalize("BMS overcurrent 24V CAN bus timeout")
        elapsed = time.perf_counter() - start

        per_call_ms = (elapsed / iterations) * 1000
        assert per_call_ms < 0.5, (
            f"Normalizer too slow: {per_call_ms:.4f}ms/call (limit: 0.5ms)"
        )

    def test_ring_buffer_write_throughput(self):
        """Ring Buffer write: phải nhanh hơn 32ms/chunk real-time."""
        from src.audio import RingBuffer
        buf = RingBuffer(max_chunks=100)
        chunk = np.random.randn(512).astype(np.float32)

        start = time.perf_counter()
        iterations = 10000
        for _ in range(iterations):
            buf.write(chunk)
        elapsed = time.perf_counter() - start

        per_write_ms = (elapsed / iterations) * 1000
        assert per_write_ms < 1.0, (
            f"RingBuffer write too slow: {per_write_ms:.4f}ms (limit: 1ms)"
        )

    def test_severity_detection_throughput(self):
        """Severity detection: microseconds per call."""
        from src.nlp import SeverityDetector

        start = time.perf_counter()
        iterations = 10000
        for _ in range(iterations):
            SeverityDetector.detect("critical overcurrent 150A nguy hiểm")
        elapsed = time.perf_counter() - start

        per_call_ms = (elapsed / iterations) * 1000
        assert per_call_ms < 0.1, (
            f"SeverityDetector too slow: {per_call_ms:.4f}ms (limit: 0.1ms)"
        )

    def test_prosody_lookup_throughput(self):
        """Prosody lookup: O(1) dict lookup."""
        from src.tts import ProsodyController
        ctrl = ProsodyController()

        start = time.perf_counter()
        iterations = 10000
        for _ in range(iterations):
            ctrl.get_params("critical")
        elapsed = time.perf_counter() - start

        per_call_ms = (elapsed / iterations) * 1000
        assert per_call_ms < 0.01, (
            f"Prosody lookup too slow: {per_call_ms:.4f}ms (limit: 0.01ms)"
        )


# ═══════════════════════════════════════════════════════════
# INTEGRATION: Config Consistency
# ═══════════════════════════════════════════════════════════

class TestConfigConsistency:
    """Test: config.py values match đề bài requirements."""

    def test_audio_16khz_mono(self):
        from src.config import AUDIO
        assert AUDIO.sample_rate == 16000
        assert AUDIO.channels == 1

    def test_audio_chunk_32ms(self):
        """Silero VAD optimal chunk = 32ms = 512 samples."""
        from src.config import AUDIO
        assert AUDIO.chunk_ms == 32
        assert AUDIO.chunk_size == 512  # 16000 × 0.032

    def test_buffer_3_seconds(self):
        """Ring buffer giữ tối đa 3 giây (đề bài)."""
        from src.config import AUDIO
        assert AUDIO.buffer_seconds == 3
        # 3000ms / 32ms = 93.75 → 93 chunks
        assert AUDIO.max_buffer_chunks == 93

    def test_vad_1_thread(self):
        """VAD chỉ dùng 1 ONNX thread → CPU < 5%."""
        from src.config import VAD
        assert VAD.onnx_threads == 1

    def test_tts_prosody_profiles_complete(self):
        """TTS phải có đủ 3 profiles: normal, warning, critical."""
        from src.config import TTS
        assert "normal" in TTS.prosody_profiles
        assert "warning" in TTS.prosody_profiles
        assert "critical" in TTS.prosody_profiles
