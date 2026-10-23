import unittest

import numpy as np

from avatars.wav2lip_avatar import blend_wav2lip_face


class FaceBlendTests(unittest.TestCase):
    def test_upper_face_is_preserved_and_mouth_is_replaced(self):
        source = np.full((256, 256, 3), 20, dtype=np.uint8)
        predicted = np.full((256, 256, 3), 220, dtype=np.uint8)
        output = blend_wav2lip_face(source, predicted, .44, .10)
        self.assertTrue(np.array_equal(output[:100], source[:100]))
        self.assertTrue(np.all(output[170:] == predicted[170:]))
        self.assertGreater(int(output[128, 0, 0]), 20)
        self.assertLess(int(output[128, 0, 0]), 220)

    def test_mismatched_prediction_is_resized(self):
        source = np.zeros((256, 256, 3), dtype=np.uint8)
        predicted = np.full((96, 96, 3), 255, dtype=np.uint8)
        output = blend_wav2lip_face(source, predicted)
        self.assertEqual(output.shape, source.shape)
        self.assertTrue(np.all(output[-20:] == 255))


if __name__ == '__main__':
    unittest.main()
