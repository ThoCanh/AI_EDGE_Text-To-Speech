"""
AlwaysOnPipeline - Kiến trúc Đa luồng Producer-Consumer.

Core Khối 2: Tách biệt hoàn toàn thu âm và nhận diện.

    Thread 1 (Producer): Mic → RingBuffer → VAD → Thread-safe Queue
    Thread 2 (Consumer): Queue → ASR → Text-Norm → TTS → Speaker

Bảo vệ 3 lớp chống tràn:
    Lớp 1: VAD Timeout (10s) → force-cut speech quá dài
    Lớp 2: Bounded Queue + Drop Frame → backpressure
    Lớp 3: CPU Throttle → adaptive sleep khi vượt ngưỡng

Thread 2 chỉ wake up khi Queue nhận được chunk có tiếng người.
"""

from __future__ import annotations

import gc
import logging
import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from ..config import AUDIO, PIPELINE, VAD as VAD_CFG
from ..audio import RingBuffer
from ..vad import SileroVADEngine
from ..asr import SenseVoiceASR
from ..tts import ValtecTTSEngine
from ..system import CPUGovernor

logger = logging.getLogger(__name__)


class AlwaysOnPipeline:
    """
    Pipeline Always-on: Producer-Consumer pattern.

    Lifecycle:
        1. __init__(): Load models vào RAM (1 lần duy nhất)
        2. start(): Khởi động Producer + Consumer threads
        3. Chạy liên tục cho đến khi gọi stop()
        4. stop(): Graceful shutdown

    CPU Budget:
        - Background (Thread 1 only): ≤ 40%
        - Active (Thread 1 + Thread 2): ≤ 70%
    """

    def __init__(
        self,
        vad_path: str,
        asr_path: str,
        tts_path: str,
        asr_tokens_path: Optional[str] = None,
    ) -> None:
        logger.info("=" * 60)
        logger.info("Initializing AlwaysOnPipeline...")
        logger.info("=" * 60)

        # ── Load models (1 lần) ─────────────────────
        self._vad = SileroVADEngine(model_path=vad_path)
        self._asr = SenseVoiceASR(
            model_path=asr_path,
            tokens_path=asr_tokens_path or "",
        )
        self._tts = ValtecTTSEngine(model_path=tts_path)

        # ── Shared resources ────────────────────────
        self._ring_buffer = RingBuffer()
        self._audio_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(
            maxsize=PIPELINE.queue_maxsize,
        )

        # ── Control flags ───────────────────────────
        self._running = threading.Event()
        self._consumer_busy = threading.Event()

        # ── CPU Governor ────────────────────────────
        self._cpu_gov = CPUGovernor()

        # ── Thread refs ─────────────────────────────
        self._producer_thread: Optional[threading.Thread] = None
        self._consumer_thread: Optional[threading.Thread] = None

        # ── Stats ───────────────────────────────────
        self._stats = {
            "utterances_processed": 0,
            "frames_dropped": 0,
            "vad_timeouts": 0,
        }

        logger.info("AlwaysOnPipeline initialized. Models in RAM.")

    # ═══════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════

    def start(self) -> None:
        """Khởi động pipeline."""
        self._running.set()
        self._cpu_gov.start()

        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="Producer-AudioVAD",
            daemon=True,
        )
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop,
            name="Consumer-ASRTTS",
            daemon=True,
        )
        self._producer_thread.start()
        self._consumer_thread.start()
        logger.info("Pipeline STARTED. Listening...")

    def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Stopping pipeline...")
        self._running.clear()

        if self._producer_thread:
            self._producer_thread.join(timeout=3.0)
        if self._consumer_thread:
            self._audio_queue.put(None)  # Sentinel
            self._consumer_thread.join(timeout=5.0)

        self._cpu_gov.stop()
        logger.info("Pipeline STOPPED. Stats: %s", self._stats)

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    # ═══════════════════════════════════════════════════════
    # THREAD 1: PRODUCER (Audio + VAD) ≤ 40% CPU
    # ═══════════════════════════════════════════════════════

    def _producer_loop(self) -> None:
        """
        Vòng lặp thu âm + VAD.

        1. Đọc chunk 32ms từ mic
        2. Ghi vào Ring Buffer (luôn ghi, kể cả silence)
        3. Chạy VAD inference
        4. Speech → accumulate chunks
        5. End-of-utterance hoặc timeout → đẩy vào Queue
        """
        chunk_size = AUDIO.chunk_size
        speech_chunks: list[np.ndarray] = []
        speech_start_time: Optional[float] = None

        try:
            stream = sd.InputStream(
                samplerate=AUDIO.sample_rate,
                channels=AUDIO.channels,
                dtype=AUDIO.dtype,
                blocksize=chunk_size,
            )
            stream.start()
            logger.info("[Producer] Mic opened (chunk=%dms).", AUDIO.chunk_ms)
        except Exception as exc:
            logger.error("[Producer] Mic open failed: %s", exc)
            return

        try:
            while self._running.is_set():
                try:
                    chunk, _ = stream.read(chunk_size)
                except Exception as exc:
                    logger.error("[Producer] Mic read error: %s", exc)
                    continue

                chunk = chunk.flatten()
                self._ring_buffer.write(chunk)

                vad_result = self._vad.process_chunk(chunk, AUDIO.chunk_ms)

                if vad_result.is_speech:
                    if speech_start_time is None:
                        speech_start_time = time.monotonic()
                        pre_roll = self._ring_buffer.read_last_n_ms(VAD_CFG.pre_roll_ms)
                        if pre_roll.size > 0:
                            speech_chunks.append(pre_roll)
                        logger.debug("[Producer] Speech started.")

                    speech_chunks.append(chunk)

                    # LỚP 1: VAD Timeout
                    elapsed = time.monotonic() - speech_start_time
                    if elapsed >= PIPELINE.vad_timeout_s:
                        logger.warning("[Producer] VAD timeout (%.1fs).", elapsed)
                        self._enqueue_speech(speech_chunks)
                        speech_chunks = []
                        speech_start_time = None
                        self._vad.reset()
                        self._stats["vad_timeouts"] += 1

                elif vad_result.is_end_of_utterance and speech_chunks:
                    logger.debug("[Producer] End of utterance.")
                    self._enqueue_speech(speech_chunks)
                    speech_chunks = []
                    speech_start_time = None

                elif not vad_result.is_speech and speech_chunks:
                    # VAD chuyển về SILENCE mà KHÔNG qua PENDING_SILENCE
                    # → speech quá ngắn (< min_speech_ms) → discard
                    logger.debug("[Producer] Short burst discarded (%d chunks).", len(speech_chunks))
                    speech_chunks = []
                    speech_start_time = None

                # LỚP 3: CPU Throttle
                self._cpu_gov.throttle_if_needed(
                    is_active=self._consumer_busy.is_set(),
                )
        finally:
            stream.stop()
            stream.close()
            logger.info("[Producer] Mic closed.")

    def _enqueue_speech(self, chunks: list[np.ndarray]) -> None:
        """
        Đẩy speech vào Queue (thread-safe).

        LỚP 2: Backpressure - drop frame cũ khi queue gần đầy.
        """
        if not chunks:
            return

        audio = np.concatenate(chunks)

        # Backpressure: drop oldest khi > 80% capacity
        threshold = int(PIPELINE.queue_maxsize * PIPELINE.drop_threshold)
        if self._audio_queue.qsize() >= threshold:
            try:
                dropped = self._audio_queue.get_nowait()
                del dropped
                self._stats["frames_dropped"] += 1
                logger.warning("[Queue] Backpressure: dropped oldest.")
            except queue.Empty:
                pass

        try:
            self._audio_queue.put(audio, timeout=0.1)
            logger.debug("[Queue] Enqueued %.1fs.", audio.size / AUDIO.sample_rate)
        except queue.Full:
            self._stats["frames_dropped"] += 1
            logger.warning("[Queue] Full: frame dropped.")
            del audio

    # ═══════════════════════════════════════════════════════
    # THREAD 2: CONSUMER (ASR + TTS) ≤ 70% CPU
    # ═══════════════════════════════════════════════════════

    def _consumer_loop(self) -> None:
        """
        Consumer: Queue → ASR → Text-Norm → TTS → Speaker.

        Thread NGỦL khi queue trống. Queue.get(timeout) → không block cứng.
        """
        logger.info("[Consumer] Waiting for speech...")

        while self._running.is_set():
            try:
                audio_data = self._audio_queue.get(
                    timeout=PIPELINE.consumer_get_timeout,
                )
            except queue.Empty:
                continue

            if audio_data is None:
                break  # Sentinel → shutdown

            self._consumer_busy.set()
            try:
                self._process_utterance(audio_data)
            except Exception as exc:
                logger.error("[Consumer] Error: %s", exc)
            finally:
                self._consumer_busy.clear()
                self._audio_queue.task_done()
                del audio_data
                gc.collect()

    def _process_utterance(self, audio: np.ndarray) -> None:
        """Xử lý 1 utterance: ASR → CPU check → TTS → Playback."""
        duration = audio.size / AUDIO.sample_rate
        logger.info("[Consumer] Processing %.1fs...", duration)
        t0 = time.monotonic()

        # ASR
        t_asr = time.monotonic()
        text = self._asr.transcribe(audio)
        asr_time = time.monotonic() - t_asr
        if not text.strip():
            logger.info("[Consumer] ASR empty. Skip.")
            return
        logger.info("[ASR] '%s' (%.2fs)", text, asr_time)

        # ── CPU Checkpoint: throttle GIỮA ASR và TTS ────
        # Đảm bảo CPU ≤ 70% trước khi bắt đầu TTS inference
        self._cpu_gov.throttle_consumer()

        # TTS (includes text-norm + prosody)
        t_tts = time.monotonic()
        tts_audio = self._tts.synthesize(text)
        tts_time = time.monotonic() - t_tts
        logger.info("[TTS] %d samples (%.2fs)", tts_audio.size, tts_time)

        # ── CPU Checkpoint: throttle SAU TTS ─────────────
        # Nhường CPU trước khi playback (I/O-bound, không cần nhiều CPU)
        self._cpu_gov.throttle_consumer()

        # Playback
        if tts_audio.size > 0:
            try:
                sd.play(tts_audio, samplerate=self._tts.sample_rate)
                sd.wait()
            except Exception as exc:
                logger.error("[Consumer] Playback error: %s", exc)

        self._stats["utterances_processed"] += 1
        logger.info(
            "[Consumer] Done in %.2fs (ASR=%.2f, TTS=%.2f). Total=%d",
            time.monotonic() - t0, asr_time, tts_time,
            self._stats["utterances_processed"],
        )

