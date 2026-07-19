from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ilids_windows.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ilids_audit.json"
)

EXPECTED_SEQUENCE_LENGTH = 32
EXPECTED_KEYPOINTS = 17


def load_metadata() -> list[dict[str, str]]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono metadanych: {METADATA_PATH}"
        )

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def validate_window(
    window_path: Path,
    metadata_row: dict[str, str],
) -> list[str]:
    errors: list[str] = []

    if not window_path.exists():
        return ["file_not_found"]

    try:
        with np.load(window_path, allow_pickle=False) as data:
            required_fields = {
                "keypoints",
                "observed_mask",
                "available_mask",
                "frame_valid_mask",
                "person_id",
                "camera_id",
                "start_frame",
                "end_frame",
            }

            missing_fields = required_fields - set(data.files)

            if missing_fields:
                errors.append(
                    "missing_fields:"
                    + ",".join(sorted(missing_fields))
                )
                return errors

            keypoints = data["keypoints"]
            observed_mask = data["observed_mask"]
            available_mask = data["available_mask"]
            frame_valid_mask = data["frame_valid_mask"]

            expected_keypoints_shape = (
                EXPECTED_SEQUENCE_LENGTH,
                EXPECTED_KEYPOINTS,
                3,
            )

            expected_point_mask_shape = (
                EXPECTED_SEQUENCE_LENGTH,
                EXPECTED_KEYPOINTS,
            )

            if keypoints.shape != expected_keypoints_shape:
                errors.append(
                    f"keypoints_shape:{keypoints.shape}"
                )

            if observed_mask.shape != expected_point_mask_shape:
                errors.append(
                    f"observed_mask_shape:{observed_mask.shape}"
                )

            if available_mask.shape != expected_point_mask_shape:
                errors.append(
                    f"available_mask_shape:{available_mask.shape}"
                )

            if frame_valid_mask.shape != (
                EXPECTED_SEQUENCE_LENGTH,
            ):
                errors.append(
                    f"frame_valid_mask_shape:{frame_valid_mask.shape}"
                )

            if keypoints.dtype != np.float32:
                errors.append(
                    f"keypoints_dtype:{keypoints.dtype}"
                )

            if not np.isfinite(keypoints).all():
                errors.append("non_finite_keypoints")

            if (
                observed_mask.shape == available_mask.shape
                and np.any(observed_mask & ~available_mask)
            ):
                errors.append(
                    "observed_not_available"
                )

            if keypoints.shape == expected_keypoints_shape:
                confidence = keypoints[:, :, 2]

                if np.any(confidence < 0.0):
                    errors.append(
                        "negative_confidence"
                    )

                if np.any(confidence > 1.0):
                    errors.append(
                        "confidence_above_one"
                    )

            stored_person_id = int(data["person_id"])
            stored_camera_id = int(data["camera_id"])

            metadata_person_id = int(
                metadata_row["person_id"]
            )

            metadata_camera_id = int(
                metadata_row["camera_id"]
            )

            if stored_person_id != metadata_person_id:
                errors.append(
                    "person_id_mismatch"
                )

            if stored_camera_id != metadata_camera_id:
                errors.append(
                    "camera_id_mismatch"
                )

            start_frame = int(data["start_frame"])
            end_frame = int(data["end_frame"])

            if end_frame - start_frame != EXPECTED_SEQUENCE_LENGTH:
                errors.append(
                    "incorrect_frame_range"
                )

    except Exception as exception:
        errors.append(
            f"load_error:{type(exception).__name__}"
        )

    return errors


def main() -> None:
    rows = load_metadata()

    if not rows:
        raise RuntimeError(
            "Plik metadanych nie zawiera żadnych buforów."
        )

    split_window_counts: Counter[str] = Counter()
    split_identities: dict[str, set[int]] = defaultdict(set)

    windows_per_identity: Counter[
        tuple[str, int]
    ] = Counter()

    cameras_per_identity: dict[
        tuple[str, int],
        set[int],
    ] = defaultdict(set)

    error_counts: Counter[str] = Counter()
    invalid_files: list[dict[str, object]] = []

    observed_ratios: list[float] = []
    available_ratios: list[float] = []
    valid_frame_ratios: list[float] = []

    print(
        f"Sprawdzanie {len(rows)} buforów..."
    )

    for index, row in enumerate(rows, start=1):
        relative_path = Path(row["path"])
        window_path = PROJECT_ROOT / relative_path

        split = row["split"]
        person_id = int(row["person_id"])
        camera_id = int(row["camera_id"])

        split_window_counts[split] += 1
        split_identities[split].add(person_id)

        identity_key = (split, person_id)

        windows_per_identity[identity_key] += 1
        cameras_per_identity[identity_key].add(camera_id)

        observed_ratios.append(
            float(row["observed_keypoint_ratio"])
        )

        available_ratios.append(
            float(row["available_keypoint_ratio"])
        )

        valid_frame_ratios.append(
            float(row["valid_frame_ratio"])
        )

        errors = validate_window(
            window_path=window_path,
            metadata_row=row,
        )

        if errors:
            invalid_files.append(
                {
                    "path": row["path"],
                    "errors": errors,
                }
            )

            error_counts.update(errors)

        if index % 500 == 0 or index == len(rows):
            print(
                f"  {index}/{len(rows)}"
            )

    identities_with_one_window = Counter()
    identities_with_less_than_four = Counter()
    identities_with_both_cameras = Counter()
    identities_with_one_camera = Counter()

    for identity_key, window_count in windows_per_identity.items():
        split, _ = identity_key
        camera_count = len(
            cameras_per_identity[identity_key]
        )

        if window_count == 1:
            identities_with_one_window[split] += 1

        if window_count < 4:
            identities_with_less_than_four[split] += 1

        if camera_count == 2:
            identities_with_both_cameras[split] += 1
        else:
            identities_with_one_camera[split] += 1

    report = {
        "total_windows": len(rows),
        "valid_files": len(rows) - len(invalid_files),
        "invalid_files": len(invalid_files),
        "split_window_counts": dict(split_window_counts),
        "split_identity_counts": {
            split: len(person_ids)
            for split, person_ids in split_identities.items()
        },
        "identities_with_one_window": dict(
            identities_with_one_window
        ),
        "identities_with_less_than_four_windows": dict(
            identities_with_less_than_four
        ),
        "identities_with_both_cameras": dict(
            identities_with_both_cameras
        ),
        "identities_with_only_one_camera": dict(
            identities_with_one_camera
        ),
        "mean_valid_frame_ratio": float(
            np.mean(valid_frame_ratios)
        ),
        "mean_observed_keypoint_ratio": float(
            np.mean(observed_ratios)
        ),
        "mean_available_keypoint_ratio": float(
            np.mean(available_ratios)
        ),
        "minimum_windows_per_identity": {
            split: min(
                count
                for (identity_split, _), count
                in windows_per_identity.items()
                if identity_split == split
            )
            for split in split_identities
        },
        "maximum_windows_per_identity": {
            split: max(
                count
                for (identity_split, _), count
                in windows_per_identity.items()
                if identity_split == split
            )
            for split in split_identities
        },
        "error_counts": dict(error_counts),
        "invalid_examples": invalid_files[:20],
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("===== AUDYT ZBIORU iLIDS-VID =====")
    print(
        f"Wszystkie bufory:             {len(rows)}"
    )
    print(
        "Poprawne pliki:               "
        f"{report['valid_files']}"
    )
    print(
        "Uszkodzone pliki:             "
        f"{report['invalid_files']}"
    )

    for split in ("train", "val", "test"):
        window_count = split_window_counts[split]
        identity_count = len(split_identities[split])

        both_cameras = identities_with_both_cameras[
            split
        ]

        one_camera = identities_with_one_camera[
            split
        ]

        less_than_four = (
            identities_with_less_than_four[split]
        )

        print()
        print(split.upper())
        print(
            f"  Bufory:                     {window_count}"
        )
        print(
            f"  Osoby z buforami:           {identity_count}"
        )
        print(
            f"  Osoby z obiema kamerami:    {both_cameras}"
        )
        print(
            f"  Osoby tylko z jedną kamerą: {one_camera}"
        )
        print(
            f"  Osoby z mniej niż 4 oknami: {less_than_four}"
        )

    print()
    print(
        "Średni udział poprawnych klatek: "
        f"{100.0 * report['mean_valid_frame_ratio']:.2f}%"
    )

    print(
        "Średni udział wykrytych punktów:  "
        f"{100.0 * report['mean_observed_keypoint_ratio']:.2f}%"
    )

    print(
        "Średni udział dostępnych punktów: "
        f"{100.0 * report['mean_available_keypoint_ratio']:.2f}%"
    )

    print()
    print(f"Raport zapisano w: {REPORT_PATH}")

    if invalid_files:
        print()
        print("Wykryto błędy w plikach.")
        print(
            "Nie przechodź jeszcze do treningu."
        )
        raise SystemExit(1)

    print()
    print("Wszystkie pliki przeszły kontrolę.")


if __name__ == "__main__":
    main()