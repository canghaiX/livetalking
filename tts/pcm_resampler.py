"""Stateful PCM resampling for streaming TTS audio."""

from __future__ import annotations

import numpy as np

try:  # Python 3.11; removed from the stdlib in Python 3.13.
    import audioop
except ImportError:  # pragma: no cover - compatibility fallback
    audioop = None


class PcmResampler:
    """Convert mono signed 16-bit PCM without resetting packet phase."""

    def __init__(self, source_rate: int, target_rate: int):
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self._state = None

    def reset(self) -> None:
        self._state = None

    def process(self, pcm_data: bytes) -> np.ndarray:
        if not pcm_data:
            return np.empty(0, dtype=np.float32)
        if audioop is not None:
            converted, self._state = audioop.ratecv(
                pcm_data, 2, 1, self.source_rate, self.target_rate, self._state
            )
            return np.frombuffer(converted, dtype=np.int16).astype(np.float32) / 32768.0

        # Python 3.13 compatibility. Production images currently use the
        # stateful audioop path above; this fallback keeps imports working.
        from scipy.signal import resample_poly
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        return resample_poly(samples, self.target_rate, self.source_rate).astype(np.float32) / 32768.0
