import unittest

import cv2
import numpy as np

from tools.normalize_avatar_appearance import (
    appearance_profile,
    apply_appearance_correction,
    correction_from_profiles,
    inner_face_mask,
    soft_subject_mask,
)


class AvatarAppearanceTests(unittest.TestCase):
    def _frame(self, value):
        frame = np.full((120, 100, 3), 145, dtype=np.uint8)
        frame[20:105, 30:72] = [value, value - 4, value - 8]
        return frame

    def test_subject_mask_preserves_background(self):
        frame = self._frame(80)
        mask = soft_subject_mask(frame)
        self.assertLess(float(mask[2, 2]), .1)
        self.assertGreater(float(mask[70, 50]), .5)

    def test_correction_matches_dark_face_without_touching_background(self):
        source = self._frame(80)
        target = self._frame(110)
        box = [30, 20, 72, 105]
        source_profile = appearance_profile([source], [box])
        target_profile = appearance_profile([target], [box])
        correction = correction_from_profiles(source_profile, target_profile)
        result = apply_appearance_correction(
            source, correction, soft_subject_mask(source), inner_face_mask(source.shape, box))
        self.assertLess(float(np.abs(result[0, 0].astype(int) - source[0, 0]).max()), 2)
        source_lab = cv2.cvtColor(source[35:85, 40:65], cv2.COLOR_BGR2LAB)
        result_lab = cv2.cvtColor(result[35:85, 40:65], cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target[35:85, 40:65], cv2.COLOR_BGR2LAB)
        source_distance = abs(float(np.median(source_lab[..., 0])) -
                              float(np.median(target_lab[..., 0])))
        result_distance = abs(float(np.median(result_lab[..., 0])) -
                              float(np.median(target_lab[..., 0])))
        self.assertLess(result_distance, source_distance)
        self.assertGreater(float(np.abs(result_lab.astype(int) - source_lab).mean()), 0)


if __name__ == '__main__':
    unittest.main()
