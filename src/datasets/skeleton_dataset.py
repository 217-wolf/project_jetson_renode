from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SEQUENCE_LENGTH = 32
NUM_KEYPOINTS = 17
NUM_FEATURES = 3


@dataclass(frozen=True)
class WindowRecord:
    path: Path
    split: str
    person_name: str
    person_id: int
    camera_id: int
    window_index: int
    start_frame: int
    end_frame: int
    valid_frame_ratio: float
    observed_keypoint_ratio: float
    available_keypoint_ratio: float


class SkeletonWindowDataset(Dataset):
    """
    Odczytuje przygotowane bufory szkieletów zapisane jako NPZ.

    Każdy element zawiera:
    - keypoints:             (32, 17, 3)
    - observed_mask:         (32, 17)
    - available_mask:        (32, 17)
    - frame_valid_mask:      (32,)
    - person_id:             oryginalne ID osoby
    - label:                 etykieta ciągła 0..N-1
    - camera_id:             1 albo 2
    """

    def __init__(
        self,
        metadata_path: str | Path,
        split: str,
        project_root: str | Path | None = None,
        minimum_valid_frame_ratio: float = 0.0,
        minimum_observed_keypoint_ratio: float = 0.0,
    ) -> None:
        super().__init__()

        self.metadata_path = Path(metadata_path).resolve()

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono pliku metadanych: "
                f"{self.metadata_path}"
            )

        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split musi być jednym z: train, val, test."
            )

        if project_root is None:
            # metadata_path:
            # projekt/data/metadata/ilids_windows.csv
            self.project_root = self.metadata_path.parents[2]
        else:
            self.project_root = Path(project_root).resolve()

        self.split = split

        self.records = self._load_records(
            minimum_valid_frame_ratio=(
                minimum_valid_frame_ratio
            ),
            minimum_observed_keypoint_ratio=(
                minimum_observed_keypoint_ratio
            ),
        )

        if not self.records:
            raise RuntimeError(
                f"Brak rekordów dla splitu: {split}"
            )

        person_ids = sorted(
            {
                record.person_id
                for record in self.records
            }
        )

        self.person_id_to_label = {
            person_id: label
            for label, person_id in enumerate(person_ids)
        }

        self.label_to_person_id = {
            label: person_id
            for person_id, label
            in self.person_id_to_label.items()
        }

        identity_to_indices: dict[
            int,
            list[int],
        ] = defaultdict(list)

        for index, record in enumerate(self.records):
            identity_to_indices[
                record.person_id
            ].append(index)

        self.identity_to_indices = dict(
            identity_to_indices
        )

    def _load_records(
        self,
        minimum_valid_frame_ratio: float,
        minimum_observed_keypoint_ratio: float,
    ) -> list[WindowRecord]:
        records: list[WindowRecord] = []

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["split"] != self.split:
                    continue

                valid_ratio = float(
                    row["valid_frame_ratio"]
                )

                observed_ratio = float(
                    row["observed_keypoint_ratio"]
                )

                if (
                    valid_ratio
                    < minimum_valid_frame_ratio
                ):
                    continue

                if (
                    observed_ratio
                    < minimum_observed_keypoint_ratio
                ):
                    continue

                relative_path = Path(row["path"])

                records.append(
                    WindowRecord(
                        path=(
                            self.project_root
                            / relative_path
                        ),
                        split=row["split"],
                        person_name=row["person_name"],
                        person_id=int(row["person_id"]),
                        camera_id=int(row["camera_id"]),
                        window_index=int(
                            row["window_index"]
                        ),
                        start_frame=int(
                            row["start_frame"]
                        ),
                        end_frame=int(
                            row["end_frame"]
                        ),
                        valid_frame_ratio=valid_ratio,
                        observed_keypoint_ratio=(
                            observed_ratio
                        ),
                        available_keypoint_ratio=float(
                            row[
                                "available_keypoint_ratio"
                            ]
                        ),
                    )
                )

        return records

    @property
    def num_identities(self) -> int:
        return len(self.identity_to_indices)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | str]:
        record = self.records[index]

        if not record.path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono bufora: {record.path}"
            )

        with np.load(
            record.path,
            allow_pickle=False,
        ) as data:
            keypoints = data["keypoints"].astype(
                np.float32,
                copy=True,
            )

            observed_mask = data[
                "observed_mask"
            ].astype(
                np.bool_,
                copy=True,
            )

            available_mask = data[
                "available_mask"
            ].astype(
                np.bool_,
                copy=True,
            )

            frame_valid_mask = data[
                "frame_valid_mask"
            ].astype(
                np.bool_,
                copy=True,
            )

        expected_keypoints_shape = (
            SEQUENCE_LENGTH,
            NUM_KEYPOINTS,
            NUM_FEATURES,
        )

        expected_point_mask_shape = (
            SEQUENCE_LENGTH,
            NUM_KEYPOINTS,
        )

        if keypoints.shape != expected_keypoints_shape:
            raise ValueError(
                f"Niepoprawny kształt keypoints "
                f"w {record.path}: {keypoints.shape}"
            )

        if observed_mask.shape != expected_point_mask_shape:
            raise ValueError(
                f"Niepoprawny observed_mask "
                f"w {record.path}: {observed_mask.shape}"
            )

        if available_mask.shape != expected_point_mask_shape:
            raise ValueError(
                f"Niepoprawny available_mask "
                f"w {record.path}: {available_mask.shape}"
            )

        if frame_valid_mask.shape != (
            SEQUENCE_LENGTH,
        ):
            raise ValueError(
                f"Niepoprawny frame_valid_mask "
                f"w {record.path}: "
                f"{frame_valid_mask.shape}"
            )

        label = self.person_id_to_label[
            record.person_id
        ]

        return {
            "keypoints": torch.from_numpy(keypoints),
            "observed_mask": torch.from_numpy(
                observed_mask
            ),
            "available_mask": torch.from_numpy(
                available_mask
            ),
            "frame_valid_mask": torch.from_numpy(
                frame_valid_mask
            ),
            "person_id": torch.tensor(
                record.person_id,
                dtype=torch.long,
            ),
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "camera_id": torch.tensor(
                record.camera_id,
                dtype=torch.long,
            ),
            "valid_frame_ratio": torch.tensor(
                record.valid_frame_ratio,
                dtype=torch.float32,
            ),
            "observed_keypoint_ratio": torch.tensor(
                record.observed_keypoint_ratio,
                dtype=torch.float32,
            ),
            "path": str(record.path),
        }