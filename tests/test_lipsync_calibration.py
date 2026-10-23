import unittest
from pathlib import Path

import wave

import numpy as np

from tools.lipsync_calibrate import calibration_plan, offsets, select_offset
from tools.musetalk_ab_test import build_report


class LipSyncCalibrationTests(unittest.TestCase):
    def test_offset_sweep_is_bounded_and_frame_aligned(self):
        values = offsets()
        self.assertEqual(values[0], -120)
        self.assertEqual(values[-1], 120)
        self.assertEqual(values, list(range(-120, 121, 20)))
        self.assertEqual(calibration_plan(Path('/tmp/serena.wav'))['max_extra_startup_ms'], 300)

    def test_measured_onset_selects_nearest_offset(self):
        path = Path('/tmp/lipsync-calibration.wav')
        samples = np.zeros(3200, dtype='<i2')
        samples[1600:] = 12000
        with wave.open(str(path), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(samples.tobytes())
        selected, measured = select_offset(100, 141)
        self.assertEqual(selected, 40)
        self.assertEqual(measured, 41)

    def test_musetalk_missing_weights_does_not_enable_production(self):
        report = build_report(Path('/tmp/does-not-exist'))
        self.assertFalse(report['ready'])
        self.assertEqual(report['production_model'], 'wav2lip')
        self.assertEqual(report['status'], 'blocked_missing_weights')


if __name__ == '__main__':
    unittest.main()
