from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets.sequence_windows import build_sequence_windows
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

MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n-pose.pt"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ilids_pose"
    / "full"
)

SEQUENCES_ROOT = OUTPUT_ROOT / "sequences"
WINDOWS_ROOT = OUTPUT_ROOT / "windows"

METADATA_ROOT = PROJECT_ROOT / "data" / "metadata"

IDENTITY_SPLIT_PATH = (
    METADATA_ROOT
    / "ilids_identity_split.csv"
)

SEQUENCE_METADATA_PATH = (
    METADATA_ROOT
    / "ilids_sequences.csv"
)

WINDOW_METADATA_PATH = (
    METADATA_ROOT
    / "ilids_windows.csv"
)


RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

CONFIDENCE_THRESHOLD = 0.25

SEQUENCE_LENGTH = 32
STRIDE = 8

MINIMUM_VALID_FRAME_RATIO = 0.65
MINIMUM_OBSERVED_KEYPOINT_RATIO = 0.45

MAX_INTERNAL_GAP = 5
MAX_EDGE_GAP = 2

YOLO_BATCH_SIZE = 32
YOLO_IMAGE_SIZE = 256
YOLO_DETECTION_THRESHOLD = 0.10


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Przygotowuje cały zbiór iLIDS-VID do treningu "
            "sieci Re-ID opartej na szkielecie."
        )
    )

    parser.add_argument(
        "--max-persons",
        type=int,
        default=None,
        help=(
            "Opcjonalne ograniczenie liczby osób. "
            "Przydatne do testu skryptu."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ponownie uruchamia YOLO także dla sekwencji, "
            "które zostały już przetworzone."
        ),
    )

    return parser.parse_args()


def extract_person_id(person_name: str) -> int:
    match = re.search(r"(\d+)$", person_name)

    if match is None:
        raise ValueError(
            f"Nie można odczytać ID z nazwy: {person_name}"
        )

    return int(match.group(1))


def find_complete_identities() -> list[str]:
    cam1_root = DATASET_ROOT / "cam1"
    cam2_root = DATASET_ROOT / "cam2"

    if not cam1_root.exists():
        raise FileNotFoundError(
            f"Nie znaleziono katalogu: {cam1_root}"
        )

    if not cam2_root.exists():
        raise FileNotFoundError(
            f"Nie znaleziono katalogu: {cam2_root}"
        )

    cam1_people = {
        path.name
        for path in cam1_root.iterdir()
        if path.is_dir()
    }

    cam2_people = {
        path.name
        for path in cam2_root.iterdir()
        if path.is_dir()
    }

    complete_people = sorted(
        cam1_people & cam2_people,
        key=extract_person_id,
    )

    if not complete_people:
        raise RuntimeError(
            "Nie znaleziono osób obecnych w obu kamerach."
        )

    return complete_people


def create_identity_split(
    identities: list[str],
) -> dict[str, str]:
    shuffled = identities.copy()

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(shuffled)

    identity_count = len(shuffled)

    train_count = int(identity_count * TRAIN_RATIO)
    val_count = int(identity_count * VAL_RATIO)

    train_people = set(shuffled[:train_count])

    val_people = set(
        shuffled[
            train_count:
            train_count + val_count
        ]
    )

    test_people = set(
        shuffled[
            train_count + val_count:
        ]
    )

    split_by_person: dict[str, str] = {}

    for person_name in identities:
        if person_name in train_people:
            split_by_person[person_name] = "train"
        elif person_name in val_people:
            split_by_person[person_name] = "val"
        elif person_name in test_people:
            split_by_person[person_name] = "test"
        else:
            raise RuntimeError(
                f"Nie przypisano podziału dla {person_name}."
            )

    return split_by_person


def save_identity_split(
    identities: list[str],
    split_by_person: dict[str, str],
) -> None:
    METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with IDENTITY_SPLIT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "person_name",
                "person_id",
                "split",
            ],
        )

        writer.writeheader()

        for person_name in identities:
            writer.writerow(
                {
                    "person_name": person_name,
                    "person_id": extract_person_id(
                        person_name
                    ),
                    "split": split_by_person[
                        person_name
                    ],
                }
            )


def extract_sequence(
    detector: PoseDetector,
    image_paths: list[Path],
    person_id: int,
    camera_id: int,
    output_path: Path,
) -> None:
    frame_count = len(image_paths)

    keypoints = np.zeros(
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

    detections = detector.predict_paths(
        image_paths=image_paths,
        batch_size=YOLO_BATCH_SIZE,
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
            continue

        keypoints[frame_index] = normalized
        masks[frame_index] = mask
        valid_frames[frame_index] = True

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_names = np.asarray(
        [path.name for path in image_paths]
    )

    np.savez_compressed(
        output_path,
        keypoints=keypoints,
        mask=masks,
        bounding_boxes=bounding_boxes,
        detection_confidence=detection_confidences,
        detected_frames=detected_frames,
        valid_frames=valid_frames,
        frame_names=frame_names,
        person_id=np.int32(person_id),
        camera_id=np.int32(camera_id),
    )


def load_sequence(
    sequence_path: Path,
) -> dict[str, np.ndarray | int]:
    with np.load(sequence_path) as data:
        return {
            "keypoints": data["keypoints"].copy(),
            "mask": data["mask"].copy(),
            "detected_frames": (
                data["detected_frames"].copy()
            ),
            "valid_frames": (
                data["valid_frames"].copy()
            ),
            "frame_names": data["frame_names"].copy(),
            "person_id": int(data["person_id"]),
            "camera_id": int(data["camera_id"]),
        }


def save_windows(
    sequence_data: dict[str, np.ndarray | int],
    split_name: str,
    person_name: str,
    camera_name: str,
) -> tuple[list[dict[str, object]], int]:
    keypoints = sequence_data["keypoints"]
    mask = sequence_data["mask"]
    valid_frames = sequence_data["valid_frames"]
    frame_names = sequence_data["frame_names"]

    assert isinstance(keypoints, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert isinstance(valid_frames, np.ndarray)
    assert isinstance(frame_names, np.ndarray)

    person_id = int(sequence_data["person_id"])
    camera_id = int(sequence_data["camera_id"])

    windows = build_sequence_windows(
        keypoints=keypoints,
        mask=mask,
        valid_frames=valid_frames,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        minimum_valid_frame_ratio=(
            MINIMUM_VALID_FRAME_RATIO
        ),
        minimum_observed_keypoint_ratio=(
            MINIMUM_OBSERVED_KEYPOINT_RATIO
        ),
        max_internal_gap=MAX_INTERNAL_GAP,
        max_edge_gap=MAX_EDGE_GAP,
    )

    camera_output_directory = (
        WINDOWS_ROOT
        / split_name
        / person_name
        / camera_name
    )

    # Usuwamy poprzednie okna tej konkretnej sekwencji.
    if camera_output_directory.exists():
        shutil.rmtree(camera_output_directory)

    camera_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_rows: list[dict[str, object]] = []

    for window_index, window in enumerate(windows):
        window_path = (
            camera_output_directory
            / f"window_{window_index:03d}.npz"
        )

        selected_frame_names = frame_names[
            window.start_frame:
            window.end_frame
        ]

        valid_ratio = float(
            np.mean(window.frame_valid_mask)
        )

        observed_ratio = float(
            np.mean(window.observed_mask)
        )

        available_ratio = float(
            np.mean(window.available_mask)
        )

        np.savez_compressed(
            window_path,
            keypoints=window.keypoints,
            observed_mask=window.observed_mask,
            available_mask=window.available_mask,
            frame_valid_mask=window.frame_valid_mask,
            frame_names=selected_frame_names,
            person_id=np.int32(person_id),
            camera_id=np.int32(camera_id),
            start_frame=np.int32(
                window.start_frame
            ),
            end_frame=np.int32(
                window.end_frame
            ),
        )

        metadata_rows.append(
            {
                "path": window_path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "split": split_name,
                "person_name": person_name,
                "person_id": person_id,
                "camera_id": camera_id,
                "window_index": window_index,
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
                "valid_frame_ratio": (
                    f"{valid_ratio:.6f}"
                ),
                "observed_keypoint_ratio": (
                    f"{observed_ratio:.6f}"
                ),
                "available_keypoint_ratio": (
                    f"{available_ratio:.6f}"
                ),
            }
        )

    return metadata_rows, len(windows)


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    arguments = parse_arguments()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {MODEL_PATH}"
        )

    identities = find_complete_identities()

    split_by_person = create_identity_split(
        identities
    )

    save_identity_split(
        identities=identities,
        split_by_person=split_by_person,
    )

    selected_identities = identities

    if arguments.max_persons is not None:
        if arguments.max_persons <= 0:
            raise ValueError(
                "--max-persons musi być większe od zera."
            )

        selected_identities = identities[
            :arguments.max_persons
        ]

    detector = PoseDetector(
        model_path=MODEL_PATH,
        device=0,
        image_size=YOLO_IMAGE_SIZE,
        detection_threshold=(
            YOLO_DETECTION_THRESHOLD
        ),
    )

    total_sequences = (
        len(selected_identities) * 2
    )

    sequence_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []

    split_window_counts: Counter[str] = Counter()
    split_sequence_counts: Counter[str] = Counter()

    sequences_without_windows = 0
    sequence_number = 0

    for person_name in selected_identities:
        person_id = extract_person_id(
            person_name
        )

        split_name = split_by_person[
            person_name
        ]

        for camera_id in (1, 2):
            sequence_number += 1

            camera_name = f"cam{camera_id}"

            source_directory = (
                DATASET_ROOT
                / camera_name
                / person_name
            )

            image_paths = sorted(
                source_directory.glob("*.png")
            )

            if not image_paths:
                print(
                    f"[{sequence_number}/{total_sequences}] "
                    f"Brak klatek: "
                    f"{camera_name}/{person_name}"
                )
                continue

            sequence_path = (
                SEQUENCES_ROOT
                / split_name
                / person_name
                / f"{camera_name}.npz"
            )

            if arguments.force or not sequence_path.exists():
                extract_sequence(
                    detector=detector,
                    image_paths=image_paths,
                    person_id=person_id,
                    camera_id=camera_id,
                    output_path=sequence_path,
                )

                source_status = "YOLO"
            else:
                source_status = "CACHE"

            sequence_data = load_sequence(
                sequence_path
            )

            detected_frames = sequence_data[
                "detected_frames"
            ]

            valid_frames = sequence_data[
                "valid_frames"
            ]

            assert isinstance(
                detected_frames,
                np.ndarray,
            )

            assert isinstance(
                valid_frames,
                np.ndarray,
            )

            detected_count = int(
                np.count_nonzero(
                    detected_frames
                )
            )

            valid_count = int(
                np.count_nonzero(
                    valid_frames
                )
            )

            new_window_rows, window_count = (
                save_windows(
                    sequence_data=sequence_data,
                    split_name=split_name,
                    person_name=person_name,
                    camera_name=camera_name,
                )
            )

            window_rows.extend(
                new_window_rows
            )

            if window_count == 0:
                sequences_without_windows += 1

            split_sequence_counts[
                split_name
            ] += 1

            split_window_counts[
                split_name
            ] += window_count

            sequence_rows.append(
                {
                    "path": sequence_path.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                    "split": split_name,
                    "person_name": person_name,
                    "person_id": person_id,
                    "camera_id": camera_id,
                    "frame_count": len(image_paths),
                    "detected_frames": detected_count,
                    "valid_frames": valid_count,
                    "valid_frame_ratio": (
                        f"{valid_count / len(image_paths):.6f}"
                    ),
                    "window_count": window_count,
                }
            )

            print(
                f"[{sequence_number:03d}/"
                f"{total_sequences:03d}] "
                f"{split_name:5s} "
                f"{person_name} "
                f"{camera_name} "
                f"{source_status:5s} | "
                f"klatki={len(image_paths):3d}, "
                f"poprawne={valid_count:3d}, "
                f"bufory={window_count:2d}"
            )

    write_csv(
        path=SEQUENCE_METADATA_PATH,
        rows=sequence_rows,
        fieldnames=[
            "path",
            "split",
            "person_name",
            "person_id",
            "camera_id",
            "frame_count",
            "detected_frames",
            "valid_frames",
            "valid_frame_ratio",
            "window_count",
        ],
    )

    write_csv(
        path=WINDOW_METADATA_PATH,
        rows=window_rows,
        fieldnames=[
            "path",
            "split",
            "person_name",
            "person_id",
            "camera_id",
            "window_index",
            "start_frame",
            "end_frame",
            "valid_frame_ratio",
            "observed_keypoint_ratio",
            "available_keypoint_ratio",
        ],
    )

    identity_split_counts = Counter(
        split_by_person[person_name]
        for person_name in identities
    )

    print()
    print("===== PODSUMOWANIE iLIDS-VID =====")
    print(
        f"Wszystkie osoby w zbiorze:   "
        f"{len(identities)}"
    )
    print(
        f"Przetworzone osoby:          "
        f"{len(selected_identities)}"
    )
    print(
        f"Przetworzone sekwencje:      "
        f"{len(sequence_rows)}"
    )
    print(
        f"Utworzone bufory:            "
        f"{len(window_rows)}"
    )
    print(
        f"Sekwencje bez buforów:       "
        f"{sequences_without_windows}"
    )

    print()
    print("Podział wszystkich tożsamości:")
    print(
        f"  train: {identity_split_counts['train']}"
    )
    print(
        f"  val:   {identity_split_counts['val']}"
    )
    print(
        f"  test:  {identity_split_counts['test']}"
    )

    print()
    print("Bufory w bieżącym uruchomieniu:")
    print(
        f"  train: {split_window_counts['train']}"
    )
    print(
        f"  val:   {split_window_counts['val']}"
    )
    print(
        f"  test:  {split_window_counts['test']}"
    )

    print()
    print(
        f"Metadane sekwencji: "
        f"{SEQUENCE_METADATA_PATH}"
    )
    print(
        f"Metadane buforów:   "
        f"{WINDOW_METADATA_PATH}"
    )
    print(
        f"Podział osób:       "
        f"{IDENTITY_SPLIT_PATH}"
    )


if __name__ == "__main__":
    main()