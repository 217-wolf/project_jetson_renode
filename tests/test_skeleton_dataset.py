import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.identity_sampler import (
    IdentityBatchSampler,
)
from src.datasets.skeleton_dataset import (
    SkeletonWindowDataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ilids_windows.csv"
)


class SkeletonDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_dataset = SkeletonWindowDataset(
            metadata_path=METADATA_PATH,
            split="train",
            project_root=PROJECT_ROOT,
        )

        cls.val_dataset = SkeletonWindowDataset(
            metadata_path=METADATA_PATH,
            split="val",
            project_root=PROJECT_ROOT,
        )

        cls.test_dataset = SkeletonWindowDataset(
            metadata_path=METADATA_PATH,
            split="test",
            project_root=PROJECT_ROOT,
        )

    def test_split_sizes(self) -> None:
        self.assertEqual(
            len(self.train_dataset),
            2188,
        )

        self.assertEqual(
            len(self.val_dataset),
            474,
        )

        self.assertEqual(
            len(self.test_dataset),
            500,
        )

    def test_item_shapes(self) -> None:
        item = self.train_dataset[0]

        self.assertEqual(
            item["keypoints"].shape,
            (32, 17, 3),
        )

        self.assertEqual(
            item["observed_mask"].shape,
            (32, 17),
        )

        self.assertEqual(
            item["available_mask"].shape,
            (32, 17),
        )

        self.assertEqual(
            item["frame_valid_mask"].shape,
            (32,),
        )

        self.assertEqual(
            item["keypoints"].dtype,
            torch.float32,
        )

        self.assertEqual(
            item["person_id"].dtype,
            torch.int64,
        )

    def test_identities_do_not_cross_splits(
        self,
    ) -> None:
        train_ids = set(
            self.train_dataset.identity_to_indices
        )

        val_ids = set(
            self.val_dataset.identity_to_indices
        )

        test_ids = set(
            self.test_dataset.identity_to_indices
        )

        self.assertTrue(
            train_ids.isdisjoint(val_ids)
        )

        self.assertTrue(
            train_ids.isdisjoint(test_ids)
        )

        self.assertTrue(
            val_ids.isdisjoint(test_ids)
        )

    def test_identity_sampler_batch(self) -> None:
        sampler = IdentityBatchSampler(
            dataset=self.train_dataset,
            identities_per_batch=8,
            samples_per_identity=4,
            batches_per_epoch=2,
            seed=42,
        )

        data_loader = DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=0,
        )

        batch = next(iter(data_loader))

        self.assertEqual(
            batch["keypoints"].shape,
            (32, 32, 17, 3),
        )

        unique_ids, counts = torch.unique(
            batch["person_id"],
            return_counts=True,
        )

        self.assertEqual(
            len(unique_ids),
            8,
        )

        self.assertTrue(
            torch.all(counts == 4)
        )


if __name__ == "__main__":
    unittest.main()