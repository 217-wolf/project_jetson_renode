from __future__ import annotations

import numpy as np


NUM_KEYPOINTS = 17

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


class InvalidSkeletonError(ValueError):
    """Szkielet nie zawiera punktów wymaganych do normalizacji."""


def normalize_skeleton(
    keypoints: np.ndarray,
    confidence_threshold: float = 0.25,
    minimum_scale: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalizuje pojedynczy szkielet YOLO Pose.

    Parametry
    ---------
    keypoints:
        Tablica o kształcie (17, 3):
        [x, y, confidence].

    confidence_threshold:
        Minimalna pewność, aby punkt uznać za widoczny.

    minimum_scale:
        Zabezpieczenie przed dzieleniem przez bardzo małą wartość.

    Zwraca
    -------
    normalized:
        Tablica (17, 3). Współrzędne są:
        - przesunięte względem środka bioder,
        - podzielone przez długość tułowia,
        - niewidoczne punkty mają współrzędne (0, 0).

    mask:
        Tablica bool o kształcie (17,).
        True oznacza punkt widoczny.
    """
    points = np.asarray(keypoints, dtype=np.float32)

    if points.shape != (NUM_KEYPOINTS, 3):
        raise ValueError(
            "Oczekiwano tablicy o kształcie "
            f"({NUM_KEYPOINTS}, 3), otrzymano {points.shape}."
        )

    xy = points[:, :2]

    confidences = np.nan_to_num(
        points[:, 2],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    confidences = np.clip(confidences, 0.0, 1.0)

    finite_coordinates = np.isfinite(xy).all(axis=1)

    visible = (
        finite_coordinates
        & (confidences >= confidence_threshold)
    )

    required_indices = np.array(
        [
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_HIP,
            RIGHT_HIP,
        ]
    )

    if not np.all(visible[required_indices]):
        raise InvalidSkeletonError(
            "Do normalizacji wymagane są oba barki i oba biodra."
        )

    shoulder_center = np.mean(
        xy[[LEFT_SHOULDER, RIGHT_SHOULDER]],
        axis=0,
    )

    hip_center = np.mean(
        xy[[LEFT_HIP, RIGHT_HIP]],
        axis=0,
    )

    torso_length = float(
        np.linalg.norm(shoulder_center - hip_center)
    )

    if not np.isfinite(torso_length) or torso_length < minimum_scale:
        raise InvalidSkeletonError(
            "Długość tułowia jest zbyt mała do normalizacji."
        )

    normalized = np.zeros(
        (NUM_KEYPOINTS, 3),
        dtype=np.float32,
    )

    normalized[:, 2] = confidences

    normalized[visible, :2] = (
        xy[visible] - hip_center
    ) / torso_length

    return normalized, visible