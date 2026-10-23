import unittest

import numpy as np

from tools.prepare_avatar_action import protected_background_shift


class BackgroundPreparationTests(unittest.TestCase):
    def test_person_pixels_and_enclosed_gray_regions_are_protected(self):
        frame = np.full((120, 120, 3), 140, dtype=np.uint8)
        frame[30:100, 35:85] = [30, 40, 65]
        frame[45:70, 45:75] = [140, 140, 140]
        result = protected_background_shift(frame, np.array([10, 8, 10]),
                                             np.array([140, 140, 140]))
        np.testing.assert_array_equal(result[30:100, 35:85], frame[30:100, 35:85])
        np.testing.assert_array_equal(result[0, 0], [150, 148, 150])


if __name__ == '__main__':
    unittest.main()
