"""
Silero VAD Engine - Voice Activity Detection siêu nhẹ.

Chạy Silero VAD qua ONNX Runtime với 1 thread duy nhất.
CPU usage < 5% trên Pi5 ở chế độ background listening.

State machine 4 trạng thái:
    SILENCE → PENDING_SPEECH → SPEECH → PENDING_SILENCE → SILENCE

Đây là core của Khối 1: lọc bỏ tiếng ồn, chỉ cho qua tiếng người.
"""

from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort

from ..config import VAD
from .types import VADState, VADResult

logger = logging.getLogger(__name__)


class SileroVADEngine:
    """
    Wrapper cho Silero VAD model (ONNX).

    Xử lý từng chunk 32ms, trả về kết quả speech/silence
    cùng với trạng thái state machine để pipeline quyết định
    khi nào bắt đầu/kết thúc ghi âm.

    Example::

        vad = SileroVADEngine("models/silero_vad.onnx")
        result = vad.process_chunk(audio_chunk_32ms)
        if result.is_speech:
            # Accumulate speech data
        if result.is_end_of_utterance:
            # Đẩy vào queue cho ASR
    """

    def __init__(self, model_path: str = VAD.model_path) -> None:
        # ONNX session - 1 thread duy nhất cho VAD
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = VAD.onnx_threads
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_opts.log_severity_level = 3  # Suppress ONNX warnings

        self._session = ort.InferenceSession(
            model_path, sess_opts, providers=["CPUExecutionProvider"],
        )

        # VAD internal hidden states (LSTM)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._sr = np.array([16000], dtype=np.int64)

        # State machine
        self._state = VADState.SILENCE
        self._speech_ms = 0
        self._silence_ms = 0

        logger.info("Silero VAD loaded: %s (threads=%d)", model_path, VAD.onnx_threads)

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    @property
    def state(self) -> VADState:
        return self._state

    def process_chunk(self, chunk: np.ndarray, chunk_ms: int = 32) -> VADResult:
        """
        Xử lý 1 chunk audio qua VAD.

        Args:
            chunk: Audio PCM float32, 16kHz, mono. Kích thước = 512 samples (32ms).
            chunk_ms: Thời lượng chunk (ms). Mặc định 32ms.

        Returns:
            VADResult chứa is_speech, state, probability, is_end_of_utterance.
        """
        speech_prob = self._infer(chunk)

        prev_state = self._state
        is_speech = self._update_state(speech_prob, chunk_ms)

        # Phát hiện end-of-utterance: PENDING_SILENCE → SILENCE
        is_eou = (
            prev_state == VADState.PENDING_SILENCE
            and self._state == VADState.SILENCE
        )

        return VADResult(
            is_speech=is_speech,
            state=self._state,
            probability=speech_prob,
            is_end_of_utterance=is_eou,
        )

    def reset(self) -> None:
        """Reset toàn bộ state - gọi sau VAD timeout hoặc force-cut."""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._state = VADState.SILENCE
        self._speech_ms = 0
        self._silence_ms = 0
        logger.debug("VAD state reset.")

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    def _infer(self, chunk: np.ndarray) -> float:
        """Chạy ONNX inference, trả về speech probability [0, 1]."""
        ort_inputs = {
            "input": chunk.reshape(1, -1).astype(np.float32),
            "h": self._h,
            "c": self._c,
            "sr": self._sr,
        }
        output, self._h, self._c = self._session.run(None, ort_inputs)
        return float(output[0][0])

    def _update_state(self, prob: float, chunk_ms: int) -> bool:
        """
        State machine transitions.

        SILENCE → PENDING_SPEECH: prob >= threshold_on
        PENDING_SPEECH → SPEECH: tích lũy đủ min_speech_ms
        SPEECH → PENDING_SILENCE: prob < threshold_off
        PENDING_SILENCE → SILENCE: silence kéo dài >= max_silence_ms
        """
        if self._state == VADState.SILENCE:
            if prob >= VAD.threshold_on:
                self._state = VADState.PENDING_SPEECH
                self._speech_ms = chunk_ms
            return False

        if self._state == VADState.PENDING_SPEECH:
            if prob >= VAD.threshold_on:
                self._speech_ms += chunk_ms
                if self._speech_ms >= VAD.min_speech_ms:
                    self._state = VADState.SPEECH
                    return True
            else:
                self._state = VADState.SILENCE
                self._speech_ms = 0
            return False

        if self._state == VADState.SPEECH:
            if prob < VAD.threshold_off:
                self._state = VADState.PENDING_SILENCE
                self._silence_ms = chunk_ms
            return True

        if self._state == VADState.PENDING_SILENCE:
            if prob >= VAD.threshold_on:
                self._state = VADState.SPEECH
                self._silence_ms = 0
                return True
            self._silence_ms += chunk_ms
            if self._silence_ms >= VAD.max_silence_ms:
                self._state = VADState.SILENCE
                self._silence_ms = 0
                return False
            return True

        return False
