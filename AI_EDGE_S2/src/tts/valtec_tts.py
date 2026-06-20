"""
Valtec-TTS Engine - Text-to-Speech với Code-switching tích hợp.

VITS2 architecture, ~74.8M parameters, zero-shot voice cloning.
RTF ~0.24 trên CPU → nhanh hơn real-time.

Tích hợp:
    - CodeSwitchNormalizer: Việt hóa text trước khi synthesize
    - SeverityDetector: Tự động chọn prosody profile
    - ProsodyController: Điều chỉnh length_scale, noise_scale
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import TTS
from ..nlp import CodeSwitchNormalizer, SeverityDetector
from .prosody import ProsodyController

logger = logging.getLogger(__name__)


class ValtecTTSEngine:
    """
    Wrapper cho Valtec-TTS model.

    Pipeline:
        1. Text normalization (code-switching Anh → Việt)
        2. Severity detection → prosody profile
        3. TTS synthesis với prosody parameters

    Example::

        tts = ValtecTTSEngine()
        audio = tts.synthesize("Phát hiện lỗi Overcurrent trên đường nguồn 24V")
    """

    def __init__(
        self,
        model_path: str = TTS.model_path,
        auto_normalize: bool = True,
    ) -> None:
        self._model_path = model_path
        self._output_sr = TTS.output_sample_rate
        self._auto_normalize = auto_normalize

        # Sub-components
        self._normalizer = CodeSwitchNormalizer()
        self._severity = SeverityDetector()
        self._prosody = ProsodyController()

        # Load TTS model
        self._model = self._load_model(model_path)
        logger.info("Valtec-TTS loaded: %s", model_path)

    def _load_model(self, model_path: str):
        """Load model: thử Valtec SDK → ONNX → dummy mode."""
        try:
            from valtec import ValtecTTS
            model = ValtecTTS(model_path=model_path)
            logger.info("Loaded via Valtec SDK.")
            return model
        except ImportError:
            pass

        try:
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 2
            sess_opts.inter_op_num_threads = 1
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            model = ort.InferenceSession(
                model_path, sess_opts, providers=["CPUExecutionProvider"],
            )
            logger.info("Loaded via ONNX Runtime (fallback).")
            return model
        except Exception:
            logger.warning("TTS model not found. Running in dummy mode.")
            return None

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def synthesize(self, text: str, severity: Optional[str] = None) -> np.ndarray:
        """
        Tổng hợp giọng nói từ text.

        Args:
            text: Câu text (có thể chứa tiếng Anh).
            severity: Override severity. None = tự detect.

        Returns:
            Audio PCM float32. Sample rate = TTS.output_sample_rate.
        """
        if not text.strip():
            return np.array([], dtype=np.float32)

        if severity is None:
            severity = self._severity.detect(text)

        if self._auto_normalize:
            text = self._normalizer.normalize(text)

        prosody = self._prosody.get_params(severity)
        logger.info("TTS: severity=%s, text='%s'", severity, text[:80])

        return self._run_inference(text, prosody)

    def synthesize_raw(
        self, text: str,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
    ) -> np.ndarray:
        """Synthesis với prosody parameters thủ công."""
        prosody = {
            "length_scale": length_scale,
            "noise_scale": noise_scale,
            "noise_scale_w": noise_scale_w,
        }
        return self._run_inference(text, prosody)

    @property
    def sample_rate(self) -> int:
        return self._output_sr

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    def _run_inference(self, text: str, prosody: dict) -> np.ndarray:
        """Chạy TTS inference. Prosody = scalar multiply → 0 overhead."""
        if self._model is None:
            logger.warning("TTS in dummy mode, returning silence.")
            return np.zeros(int(self._output_sr * 0.5), dtype=np.float32)

        try:
            if hasattr(self._model, "synthesize"):
                audio = self._model.synthesize(
                    text,
                    length_scale=prosody["length_scale"],
                    noise_scale=prosody["noise_scale"],
                    noise_scale_w=prosody["noise_scale_w"],
                )
                if isinstance(audio, np.ndarray):
                    return audio.astype(np.float32)
                return np.array(audio, dtype=np.float32)

            logger.warning("Direct ONNX TTS not fully implemented.")
            return np.zeros(int(self._output_sr * 0.5), dtype=np.float32)
        except Exception as exc:
            logger.error("TTS inference failed: %s", exc)
            return np.zeros(int(self._output_sr * 0.5), dtype=np.float32)
