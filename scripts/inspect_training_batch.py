from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets.identity_sampler import (
    IdentityBatchSampler,
)
from src.datasets.skeleton_dataset import (
    SkeletonWindowDataset,
)


METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ilids_windows.csv"
)


def main() -> None:
    dataset = SkeletonWindowDataset(
        metadata_path=METADATA_PATH,
        split="train",
        project_root=PROJECT_ROOT,
    )

    sampler = IdentityBatchSampler(
        dataset=dataset,
        identities_per_batch=8,
        samples_per_identity=4,
        batches_per_epoch=1,
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(loader))

    unique_ids, counts = torch.unique(
        batch["person_id"],
        return_counts=True,
    )

    print("===== DATASET =====")
    print(f"Bufory treningowe: {len(dataset)}")
    print(f"Osoby z buforami:  {dataset.num_identities}")
    print()

    print("===== BATCH =====")
    print(
        f"Keypoints:       "
        f"{tuple(batch['keypoints'].shape)}"
    )

    print(
        f"Observed mask:   "
        f"{tuple(batch['observed_mask'].shape)}"
    )

    print(
        f"Available mask:  "
        f"{tuple(batch['available_mask'].shape)}"
    )

    print(
        f"Frame mask:      "
        f"{tuple(batch['frame_valid_mask'].shape)}"
    )

    print(
        f"Liczba osób:     {len(unique_ids)}"
    )

    print(
        f"Próbek na osobę: {counts.tolist()}"
    )

    print(
        f"Person IDs:      {unique_ids.tolist()}"
    )

    print(
        "Średni udział poprawnych klatek: "
        f"{100.0 * batch['valid_frame_ratio'].mean().item():.2f}%"
    )

    print(
        "Średni udział wykrytych punktów: "
        f"{100.0 * batch['observed_keypoint_ratio'].mean().item():.2f}%"
    )


if __name__ == "__main__":
    main()