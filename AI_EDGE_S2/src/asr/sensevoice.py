"""
SenseVoiceSmall ASR Engine - Nhận dạng giọng nói đa ngôn ngữ.

Sử dụng sherpa-onnx để chạy SenseVoiceSmall INT8 trên ARM64.
234M parameters, non-autoregressive → latency cực thấp.

Hỗ trợ: Chinese, English, Japanese, Korean, Cantonese.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import ASR, AUDIO

logger = logging.getLogger(__name__)


class SenseVoiceASR:
    """
    Wrapper cho SenseVoiceSmall model qua sherpa-onnx.

    Model được load 1 lần duy nhất trong __init__().
    Mọi inference đều qua RAM pointer, không file I/O.

    Example::

        asr = SenseVoiceASR()
        text = asr.transcribe(audio_float32_16khz)
    """

    def __init__(
        self,
        model_path: str = ASR.model_path,
        tokens_path: str = ASR.tokens_path,
        num_threads: int = ASR.num_threads,
    ) -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ImportError(
                "sherpa-onnx is required. Install: pip install sherpa-onnx"
            ) from exc

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            use_itn=True,
            debug=False,
            decoding_method=ASR.decoding_method,
        )
        self._sample_rate = AUDIO.sample_rate
        logger.info("SenseVoiceSmall loaded: %s (threads=%d)", model_path, num_threads)

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Chuyển đổi audio PCM thành text.

        Args:
            audio: Mảng numpy float32, 16kHz, mono (truyền qua RAM).

        Returns:
            Chuỗi text nhận dạng được. Rỗng nếu không nhận ra gì.
        """
        if audio.size == 0:
            return ""

        audio = audio.astype(np.float32).flatten()

        stream = self._recognizer.create_stream()
        stream.accept_waveform(self._sample_rate, audio)
        self._recognizer.decode_stream(stream)

        text = stream.result.text.strip()
        logger.debug("ASR result: '%s' (samples=%d)", text, audio.size)
        return text

    def transcribe_with_language(
        self, audio: np.ndarray,
    ) -> tuple[str, Optional[str]]:
        """Nhận dạng audio và trả về cả detected language."""
        if audio.size == 0:
            return "", None

        audio = audio.astype(np.float32).flatten()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self._sample_rate, audio)
        self._recognizer.decode_stream(stream)

        result = stream.result
        return result.text.strip(), getattr(result, "lang", None)
