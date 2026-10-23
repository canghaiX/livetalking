import unittest

from tools.align_avatar_video import alignment_transform, box_stats, reference_target


class AvatarAlignmentTests(unittest.TestCase):
    def test_reference_target_uses_median_face_geometry(self):
        target = reference_target([
            [[292, 165, 479, 393], [293, 166, 480, 394]],
            [[296, 169, 477, 400], [295, 170, 478, 401]],
        ])
        self.assertEqual(target, (386.25, 282.25, 184.5, 229.5))

    def test_fixed_transform_maps_source_center_and_width(self):
        matrix, scale = alignment_transform((381.5, 284.5, 194.0, 253.0),
                                             (389.5, 283.5, 184.0, 229.5))
        mapped = matrix @ [381.5, 284.5, 1.0]
        self.assertAlmostEqual(scale, 184.0 / 194.0)
        self.assertAlmostEqual(float(mapped[0]), 389.5, places=4)
        self.assertAlmostEqual(float(mapped[1]), 283.5, places=4)

    def test_box_stats_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            box_stats([])

    def test_aligned_source_geometry_matches_canonical_target(self):
        source = (389.5, 283.25, 185.0, 240.5)
        target = (389.5, 283.5, 184.0, 229.5)
        matrix, scale = alignment_transform(source, target)
        mapped = matrix @ [source[0], source[1], 1.0]
        self.assertAlmostEqual(scale, 184.0 / 185.0)
        self.assertAlmostEqual(float(mapped[0]), target[0], places=4)
        self.assertAlmostEqual(float(mapped[1]), target[1], places=4)


if __name__ == '__main__':
    unittest.main()
