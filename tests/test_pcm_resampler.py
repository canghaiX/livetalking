import importlib.util
from pathlib import Path
import unittest

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "pcm_resampler", Path(__file__).parents[1] / "tts" / "pcm_resampler.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
PcmResampler = _MODULE.PcmResampler


class PcmResamplerTest(unittest.TestCase):
    def test_split_packets_keep_the_same_audio_clock(self):
        source = (
            np.sin(2 * np.pi * 440 * np.arange(24000) / 24000) * 0.4
        ).astype(np.float32)
        pcm = (source * 32767).astype(np.int16).tobytes()

        whole = PcmResampler(24000, 16000).process(pcm)
        streaming = PcmResampler(24000, 16000)
        chunks = [
            streaming.process(pcm[offset:offset + 1554])
            for offset in range(0, len(pcm), 1554)
        ]
        split = np.concatenate(chunks)

        self.assertEqual(len(whole), 16000)
        self.assertEqual(len(split), len(whole))
        np.testing.assert_array_equal(split, whole)

    def test_reset_starts_a_new_timeline(self):
        pcm = (np.ones(2400, dtype=np.int16) * 1000).tobytes()
        resampler = PcmResampler(24000, 16000)
        first = resampler.process(pcm)
        resampler.reset()
        second = resampler.process(pcm)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
