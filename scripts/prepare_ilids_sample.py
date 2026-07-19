from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.pose.detector import PoseDetector
from src.pose.normalization import (
    InvalidSkeletonError,
    normalize_skeleton,
)


DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ilids_vid"
    / "i-LIDS-VID"
    / "sequences"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolo11n-pose.pt"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ilids_pose"
    / "sample"
)

CAMERA_NAME = "cam1"
PERSON_NAME = "person001"

CONFIDENCE_THRESHOLD = 0.25


def main() -> None:
    sequence_directory = (
        DATASET_ROOT
        / CAMERA_NAME
        / PERSON_NAME
    )

    if not sequence_directory.exists():
        raise FileNotFoundError(
            f"Nie znaleziono sekwencji: {sequence_directory}"
        )

    image_paths = sorted(
        sequence_directory.glob("*.png")
    )

    if not image_paths:
        raise FileNotFoundError(
            f"Brak klatek PNG w: {sequence_directory}"
        )

    print(f"Sekwencja: {CAMERA_NAME}/{PERSON_NAME}")
    print(f"Liczba klatek: {len(image_paths)}")
    print("Uruchamianie ekstrakcji pozy...")

    detector = PoseDetector(
        model_path=MODEL_PATH,
        device=0,
        image_size=256,
        detection_threshold=0.10,
    )

    frame_count = len(image_paths)

    normalized_keypoints = np.zeros(
        (frame_count, 17, 3),
        dtype=np.float32,
    )

    masks = np.zeros(
        (frame_count, 17),
        dtype=np.bool_,
    )

    bounding_boxes = np.zeros(
        (frame_count, 4),
        dtype=np.float32,
    )

    detection_confidences = np.zeros(
        frame_count,
        dtype=np.float32,
    )

    detected_frames = np.zeros(
        frame_count,
        dtype=np.bool_,
    )

    valid_frames = np.zeros(
        frame_count,
        dtype=np.bool_,
    )

    invalid_skeleton_count = 0

    detections = detector.predict_paths(
        image_paths=image_paths,
        batch_size=32,
    )

    for frame_index, detection in enumerate(detections):
        if detection is None:
            continue

        detected_frames[frame_index] = True
        bounding_boxes[frame_index] = detection.bbox

        detection_confidences[frame_index] = (
            detection.detection_confidence
        )

        try:
            normalized, mask = normalize_skeleton(
                detection.keypoints,
                confidence_threshold=CONFIDENCE_THRESHOLD,
            )
        except InvalidSkeletonError:
            invalid_skeleton_count += 1
            continue

        normalized_keypoints[frame_index] = normalized
        masks[frame_index] = mask
        valid_frames[frame_index] = True

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIRECTORY
        / f"{CAMERA_NAME}_{PERSON_NAME}.npz"
    )

    frame_names = np.array(
        [path.name for path in image_paths]
    )

    np.savez_compressed(
        output_path,
        keypoints=normalized_keypoints,
        mask=masks,
        bounding_boxes=bounding_boxes,
        detection_confidence=detection_confidences,
        detected_frames=detected_frames,
        valid_frames=valid_frames,
        frame_names=frame_names,
        person_id=np.int32(1),
        camera_id=np.int32(1),
    )

    detected_count = int(
        np.count_nonzero(detected_frames)
    )

    valid_count = int(
        np.count_nonzero(valid_frames)
    )

    print()
    print("===== WYNIK PRZETWARZANIA =====")
    print(f"Wszystkie klatki:        {frame_count}")
    print(f"Klatki z detekcją:       {detected_count}")
    print(f"Klatki po normalizacji:  {valid_count}")
    print(f"Odrzucone szkielety:     {invalid_skeleton_count}")
    print(
        "Brak detekcji:           "
        f"{frame_count - detected_count}"
    )

    if frame_count > 0:
        valid_percentage = (
            100.0 * valid_count / frame_count
        )

        print(
            f"Procent poprawnych:       "
            f"{valid_percentage:.2f}%"
        )

    print()
    print(f"Zapisano: {output_path}")


if __name__ == "__main__":
    main()