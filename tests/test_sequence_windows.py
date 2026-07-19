import unittest

import numpy as np

from src.datasets.sequence_windows import (
    build_sequence_windows,
    fill_short_gaps,
)


class SequenceWindowTests(unittest.TestCase):
    def test_short_gap_is_interpolated(self) -> None:
        keypoints = np.zeros(
            (6, 17, 3),
            dtype=np.float32,
        )

        mask = np.zeros(
            (6, 17),
            dtype=np.bool_,
        )

        keypoints[1, 5] = [0.0, 0.0, 0.9]
        keypoints[4, 5] = [3.0, 6.0, 0.9]

        mask[1, 5] = True
        mask[4, 5] = True

        filled, observed, available = fill_short_gaps(
            keypoints,
            mask,
            max_internal_gap=3,
            max_edge_gap=0,
        )

        np.testing.assert_allclose(
            filled[2, 5, :2],
            np.array([1.0, 2.0]),
            atol=1e-6,
        )

        np.testing.assert_allclose(
            filled[3, 5, :2],
            np.array([2.0, 4.0]),
            atol=1e-6,
        )

        self.assertFalse(observed[2, 5])
        self.assertTrue(available[2, 5])

        self.assertEqual(
            float(filled[2, 5, 2]),
            0.0,
        )

    def test_long_gap_is_not_interpolated(self) -> None:
        keypoints = np.zeros(
            (10, 17, 3),
            dtype=np.float32,
        )

        mask = np.zeros(
            (10, 17),
            dtype=np.bool_,
        )

        keypoints[1, 5] = [1.0, 1.0, 0.9]
        keypoints[8, 5] = [8.0, 8.0, 0.9]

        mask[1, 5] = True
        mask[8, 5] = True

        filled, _, available = fill_short_gaps(
            keypoints,
            mask,
            max_internal_gap=3,
            max_edge_gap=0,
        )

        np.testing.assert_allclose(
            filled[4, 5, :2],
            np.array([0.0, 0.0]),
        )

        self.assertFalse(available[4, 5])

    def test_windows_have_correct_shape(self) -> None:
        frame_count = 40

        keypoints = np.ones(
            (frame_count, 17, 3),
            dtype=np.float32,
        )

        mask = np.ones(
            (frame_count, 17),
            dtype=np.bool_,
        )

        valid_frames = np.ones(
            frame_count,
            dtype=np.bool_,
        )

        windows = build_sequence_windows(
            keypoints=keypoints,
            mask=mask,
            valid_frames=valid_frames,
            sequence_length=16,
            stride=8,
        )

        self.assertEqual(len(windows), 4)

        self.assertEqual(
            windows[0].keypoints.shape,
            (16, 17, 3),
        )

        self.assertEqual(
            windows[0].observed_mask.shape,
            (16, 17),
        )

        self.assertEqual(
            windows[0].start_frame,
            0,
        )

        self.assertEqual(
            windows[1].start_frame,
            8,
        )

    def test_window_with_too_few_valid_frames_is_rejected(
        self,
    ) -> None:
        keypoints = np.ones(
            (32, 17, 3),
            dtype=np.float32,
        )

        mask = np.ones(
            (32, 17),
            dtype=np.bool_,
        )

        valid_frames = np.zeros(
            32,
            dtype=np.bool_,
        )

        valid_frames[:10] = True

        windows = build_sequence_windows(
            keypoints=keypoints,
            mask=mask,
            valid_frames=valid_frames,
            sequence_length=32,
            stride=8,
            minimum_valid_frame_ratio=0.65,
        )

        self.assertEqual(len(windows), 0)


if __name__ == "__main__":
    unittest.main()