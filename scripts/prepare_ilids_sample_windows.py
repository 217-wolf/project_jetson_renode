from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets.sequence_windows import (
    build_sequence_windows,
)


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ilids_pose"
    / "sample"
    / "cam1_person001.npz"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ilids_pose"
    / "sample"
    / "windows"
)

SEQUENCE_LENGTH = 32
STRIDE = 8


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {INPUT_PATH}"
        )

    with np.load(INPUT_PATH) as data:
        keypoints = data["keypoints"]
        mask = data["mask"]
        valid_frames = data["valid_frames"]
        frame_names = data["frame_names"]

        person_id = int(data["person_id"])
        camera_id = int(data["camera_id"])

    windows = build_sequence_windows(
        keypoints=keypoints,
        mask=mask,
        valid_frames=valid_frames,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        minimum_valid_frame_ratio=0.65,
        minimum_observed_keypoint_ratio=0.45,
        max_internal_gap=5,
        max_edge_gap=2,
    )

    if OUTPUT_DIRECTORY.exists():
        shutil.rmtree(OUTPUT_DIRECTORY)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_ratios: list[float] = []
    observed_ratios: list[float] = []

    for window_index, window in enumerate(windows):
        output_path = (
            OUTPUT_DIRECTORY
            / f"cam1_person001_window_{window_index:03d}.npz"
        )

        selected_frame_names = frame_names[
            window.start_frame:window.end_frame
        ]

        valid_ratio = float(
            np.mean(window.frame_valid_mask)
        )

        observed_ratio = float(
            np.mean(window.observed_mask)
        )

        valid_ratios.append(valid_ratio)
        observed_ratios.append(observed_ratio)

        np.savez_compressed(
            output_path,
            keypoints=window.keypoints,
            observed_mask=window.observed_mask,
            available_mask=window.available_mask,
            frame_valid_mask=window.frame_valid_mask,
            frame_names=selected_frame_names,
            person_id=np.int32(person_id),
            camera_id=np.int32(camera_id),
            start_frame=np.int32(window.start_frame),
            end_frame=np.int32(window.end_frame),
        )

    print("===== BUFOROWANIE PRÓBNEJ SEKWENCJI =====")
    print(f"Plik wejściowy:          {INPUT_PATH}")
    print(f"Liczba klatek:           {len(keypoints)}")
    print(f"Długość bufora:          {SEQUENCE_LENGTH}")
    print(f"Przesunięcie okna:       {STRIDE}")
    print(f"Utworzone bufory:        {len(windows)}")

    if windows:
        print(
            "Średni udział poprawnych "
            f"klatek:                 "
            f"{100.0 * np.mean(valid_ratios):.2f}%"
        )

        print(
            "Średni udział wykrytych "
            f"punktów:                "
            f"{100.0 * np.mean(observed_ratios):.2f}%"
        )

    print(f"Zapisano w:              {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()