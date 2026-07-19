import unittest

import numpy as np

from src.pose.normalization import (
    InvalidSkeletonError,
    normalize_skeleton,
)


def create_example_skeleton() -> np.ndarray:
    keypoints = np.zeros((17, 3), dtype=np.float32)

    # Domyślnie wszystkie punkty są widoczne.
    keypoints[:, 2] = 0.9

    keypoints[0, :2] = [0.0, -2.4]   # nos

    keypoints[5, :2] = [-0.5, -1.0]  # lewy bark
    keypoints[6, :2] = [0.5, -1.0]   # prawy bark

    keypoints[7, :2] = [-0.8, -0.3]
    keypoints[8, :2] = [0.8, -0.3]

    keypoints[9, :2] = [-0.9, 0.4]
    keypoints[10, :2] = [0.9, 0.4]

    keypoints[11, :2] = [-0.4, 0.0]  # lewe biodro
    keypoints[12, :2] = [0.4, 0.0]   # prawe biodro

    keypoints[13, :2] = [-0.4, 1.0]
    keypoints[14, :2] = [0.4, 1.0]

    keypoints[15, :2] = [-0.4, 2.0]
    keypoints[16, :2] = [0.4, 2.0]

    return keypoints


class NormalizationTests(unittest.TestCase):
    def test_hip_center_becomes_zero(self) -> None:
        skeleton = create_example_skeleton()

        normalized, mask = normalize_skeleton(skeleton)

        hip_center = np.mean(
            normalized[[11, 12], :2],
            axis=0,
        )

        np.testing.assert_allclose(
            hip_center,
            np.array([0.0, 0.0]),
            atol=1e-6,
        )

        self.assertTrue(mask[11])
        self.assertTrue(mask[12])

    def test_translation_and_scale_invariance(self) -> None:
        original = create_example_skeleton()

        transformed = original.copy()
        transformed[:, :2] = (
            original[:, :2] * 3.7
            + np.array([420.0, 120.0])
        )

        normalized_original, mask_original = normalize_skeleton(
            original
        )

        normalized_transformed, mask_transformed = normalize_skeleton(
            transformed
        )

        np.testing.assert_array_equal(
            mask_original,
            mask_transformed,
        )

        np.testing.assert_allclose(
            normalized_original,
            normalized_transformed,
            atol=1e-5,
        )

    def test_invisible_point_is_zeroed(self) -> None:
        skeleton = create_example_skeleton()

        skeleton[9] = [999.0, 999.0, 0.1]

        normalized, mask = normalize_skeleton(skeleton)

        self.assertFalse(mask[9])

        np.testing.assert_allclose(
            normalized[9, :2],
            np.array([0.0, 0.0]),
            atol=1e-6,
        )

        self.assertAlmostEqual(
            float(normalized[9, 2]),
            0.1,
            places=5,
        )

    def test_missing_hip_is_rejected(self) -> None:
        skeleton = create_example_skeleton()

        skeleton[11, 2] = 0.0

        with self.assertRaises(InvalidSkeletonError):
            normalize_skeleton(skeleton)


if __name__ == "__main__":
    unittest.main()