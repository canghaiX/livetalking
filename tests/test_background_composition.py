import unittest

import cv2
import numpy as np

from tools.compose_avatar_background import _background_plate, _subject_mask


class BackgroundCompositionTests(unittest.TestCase):
    def test_subject_mask_keeps_dark_presenter_and_excludes_border(self):
        frame = np.full((120, 100, 3), 145, dtype=np.uint8)
        frame[20:115, 35:70] = [30, 40, 70]
        mask = _subject_mask(frame)
        self.assertGreater(float(mask[70, 50]), .5)
        self.assertLess(float(mask[2, 2]), .1)

    def test_background_plate_is_stable_and_gray(self):
        frame = np.full((120, 100, 3), [148, 139, 136], dtype=np.uint8)
        path = '/tmp/avatar-background-test.png'
        cv2.imwrite(path, frame)
        plate, color = _background_plate(path)
        self.assertEqual(plate.shape, frame.shape)
        self.assertLess(float(np.abs(plate.mean(axis=(0, 1)) - color).max()), 3)


if __name__ == '__main__':
    unittest.main()
