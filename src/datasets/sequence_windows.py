from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


NUM_KEYPOINTS = 17


@dataclass(frozen=True)
class SequenceWindow:
    """
    Jedna próbka przeznaczona dla sieci neuronowej.
    """

    keypoints: np.ndarray
    observed_mask: np.ndarray
    available_mask: np.ndarray
    frame_valid_mask: np.ndarray
    start_frame: int
    end_frame: int


def _validate_inputs(
    keypoints: np.ndarray,
    mask: np.ndarray,
    valid_frames: np.ndarray,
) -> None:
    if keypoints.ndim != 3:
        raise ValueError(
            f"keypoints musi mieć 3 wymiary, otrzymano {keypoints.shape}."
        )

    if keypoints.shape[1:] != (NUM_KEYPOINTS, 3):
        raise ValueError(
            "Oczekiwano keypoints o kształcie "
            f"(T, {NUM_KEYPOINTS}, 3), otrzymano {keypoints.shape}."
        )

    expected_mask_shape = (
        keypoints.shape[0],
        NUM_KEYPOINTS,
    )

    if mask.shape != expected_mask_shape:
        raise ValueError(
            f"Niepoprawny kształt maski: {mask.shape}. "
            f"Oczekiwano {expected_mask_shape}."
        )

    if valid_frames.shape != (keypoints.shape[0],):
        raise ValueError(
            f"Niepoprawny valid_frames: {valid_frames.shape}."
        )


def fill_short_gaps(
    keypoints: np.ndarray,
    mask: np.ndarray,
    max_internal_gap: int = 5,
    max_edge_gap: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uzupełnia krótkie braki współrzędnych punktów.

    Braki pomiędzy dwoma znanymi punktami są interpolowane liniowo.
    Krótkie braki na początku lub końcu są uzupełniane najbliższą
    dostępną wartością.

    Confidence dla uzupełnionego punktu pozostaje równy 0.
    Dzięki temu sieć wie, że współrzędne zostały oszacowane.

    Zwraca
    -------
    filled_keypoints:
        Punkty po uzupełnieniu krótkich braków.

    observed_mask:
        True tylko dla punktów faktycznie wykrytych przez YOLO.

    available_mask:
        True dla punktów wykrytych lub uzupełnionych.
    """
    points = np.asarray(
        keypoints,
        dtype=np.float32,
    ).copy()

    observed_mask = np.asarray(
        mask,
        dtype=np.bool_,
    ).copy()

    if points.ndim != 3 or points.shape[1:] != (NUM_KEYPOINTS, 3):
        raise ValueError(
            "Oczekiwano keypoints o kształcie (T, 17, 3)."
        )

    if observed_mask.shape != points.shape[:2]:
        raise ValueError(
            "Maska musi mieć kształt (T, 17)."
        )

    finite_xy = np.isfinite(points[:, :, :2]).all(axis=2)
    finite_confidence = np.isfinite(points[:, :, 2])

    observed_mask &= finite_xy
    observed_mask &= finite_confidence

    points = np.nan_to_num(
        points,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    points[:, :, 2] = np.clip(
        points[:, :, 2],
        0.0,
        1.0,
    )

    # Punkty niewykryte zaczynają jako zera.
    points[~observed_mask, :2] = 0.0
    points[~observed_mask, 2] = 0.0

    available_mask = observed_mask.copy()

    frame_count = points.shape[0]

    for keypoint_index in range(NUM_KEYPOINTS):
        observed_indices = np.flatnonzero(
            observed_mask[:, keypoint_index]
        )

        if len(observed_indices) == 0:
            continue

        first_index = int(observed_indices[0])
        last_index = int(observed_indices[-1])

        # Krótki brak na początku.
        if 0 < first_index <= max_edge_gap:
            points[:first_index, keypoint_index, :2] = (
                points[first_index, keypoint_index, :2]
            )

            available_mask[
                :first_index,
                keypoint_index,
            ] = True

        # Krótki brak na końcu.
        trailing_gap = frame_count - last_index - 1

        if 0 < trailing_gap <= max_edge_gap:
            points[
                last_index + 1:,
                keypoint_index,
                :2,
            ] = points[last_index, keypoint_index, :2]

            available_mask[
                last_index + 1:,
                keypoint_index,
            ] = True

        # Braki pomiędzy znanymi pomiarami.
        for left_index, right_index in zip(
            observed_indices[:-1],
            observed_indices[1:],
        ):
            left_index = int(left_index)
            right_index = int(right_index)

            gap_length = right_index - left_index - 1

            if gap_length <= 0:
                continue

            if gap_length > max_internal_gap:
                continue

            left_xy = points[
                left_index,
                keypoint_index,
                :2,
            ]

            right_xy = points[
                right_index,
                keypoint_index,
                :2,
            ]

            for frame_index in range(
                left_index + 1,
                right_index,
            ):
                alpha = (
                    frame_index - left_index
                ) / (
                    right_index - left_index
                )

                points[
                    frame_index,
                    keypoint_index,
                    :2,
                ] = (
                    (1.0 - alpha) * left_xy
                    + alpha * right_xy
                )

                available_mask[
                    frame_index,
                    keypoint_index,
                ] = True

    return points, observed_mask, available_mask


def build_sequence_windows(
    keypoints: np.ndarray,
    mask: np.ndarray,
    valid_frames: np.ndarray,
    sequence_length: int = 32,
    stride: int = 8,
    minimum_valid_frame_ratio: float = 0.65,
    minimum_observed_keypoint_ratio: float = 0.45,
    max_internal_gap: int = 5,
    max_edge_gap: int = 2,
) -> list[SequenceWindow]:
    """
    Tworzy przesuwane okna o stałej długości.

    Okno jest zachowywane tylko wtedy, gdy ma dostatecznie dużo:
    - poprawnych klatek,
    - rzeczywiście zaobserwowanych punktów.
    """
    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    mask = np.asarray(
        mask,
        dtype=np.bool_,
    )

    valid_frames = np.asarray(
        valid_frames,
        dtype=np.bool_,
    )

    _validate_inputs(
        keypoints,
        mask,
        valid_frames,
    )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length musi być większe od zera."
        )

    if stride <= 0:
        raise ValueError(
            "stride musi być większe od zera."
        )

    if not 0.0 <= minimum_valid_frame_ratio <= 1.0:
        raise ValueError(
            "minimum_valid_frame_ratio musi należeć do [0, 1]."
        )

    if not 0.0 <= minimum_observed_keypoint_ratio <= 1.0:
        raise ValueError(
            "minimum_observed_keypoint_ratio musi należeć do [0, 1]."
        )

    frame_count = keypoints.shape[0]

    if frame_count < sequence_length:
        return []

    (
        filled_keypoints,
        observed_mask,
        available_mask,
    ) = fill_short_gaps(
        keypoints=keypoints,
        mask=mask,
        max_internal_gap=max_internal_gap,
        max_edge_gap=max_edge_gap,
    )

    minimum_valid_frames = math.ceil(
        sequence_length
        * minimum_valid_frame_ratio
    )

    minimum_observed_points = math.ceil(
        sequence_length
        * NUM_KEYPOINTS
        * minimum_observed_keypoint_ratio
    )

    windows: list[SequenceWindow] = []

    last_start = frame_count - sequence_length

    for start_frame in range(
        0,
        last_start + 1,
        stride,
    ):
        end_frame = start_frame + sequence_length

        window_valid_frames = valid_frames[
            start_frame:end_frame
        ]

        window_observed_mask = observed_mask[
            start_frame:end_frame
        ]

        valid_frame_count = int(
            np.count_nonzero(window_valid_frames)
        )

        observed_point_count = int(
            np.count_nonzero(window_observed_mask)
        )

        if valid_frame_count < minimum_valid_frames:
            continue

        if observed_point_count < minimum_observed_points:
            continue

        windows.append(
            SequenceWindow(
                keypoints=filled_keypoints[
                    start_frame:end_frame
                ].copy(),
                observed_mask=window_observed_mask.copy(),
                available_mask=available_mask[
                    start_frame:end_frame
                ].copy(),
                frame_valid_mask=window_valid_frames.copy(),
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )

    return windows