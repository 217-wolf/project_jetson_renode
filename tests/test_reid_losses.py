import unittest

import torch

from src.reid.losses import (
    ReIDLoss,
    batch_hard_cosine_triplet_loss,
    pairwise_cosine_distance,
)


class ReIDLossTests(unittest.TestCase):
    def test_pairwise_distance_properties(self) -> None:
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        distances = pairwise_cosine_distance(
            embeddings
        )

        self.assertEqual(
            distances.shape,
            (3, 3),
        )

        torch.testing.assert_close(
            distances,
            distances.T,
        )

        torch.testing.assert_close(
            torch.diagonal(distances),
            torch.zeros(3),
        )

        self.assertAlmostEqual(
            float(distances[0, 1]),
            1.0,
            places=5,
        )

        self.assertAlmostEqual(
            float(distances[0, 2]),
            2.0,
            places=5,
        )

    def test_well_separated_embeddings_have_zero_loss(
        self,
    ) -> None:
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.99, 0.10],
                [-1.0, 0.0],
                [-0.99, -0.10],
            ],
            dtype=torch.float32,
        )

        labels = torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.long,
        )

        result = batch_hard_cosine_triplet_loss(
            embeddings=embeddings,
            labels=labels,
            margin=0.25,
        )

        self.assertAlmostEqual(
            float(result.loss),
            0.0,
            places=5,
        )

        self.assertEqual(
            result.valid_anchor_count,
            4,
        )

    def test_bad_embeddings_create_active_loss(
        self,
    ) -> None:
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.95, 0.05],
                [-0.95, -0.05],
            ],
            dtype=torch.float32,
        )

        labels = torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.long,
        )

        result = batch_hard_cosine_triplet_loss(
            embeddings=embeddings,
            labels=labels,
            margin=0.25,
        )

        self.assertGreater(
            float(result.loss),
            0.0,
        )

        self.assertGreater(
            float(result.active_fraction),
            0.0,
        )

    def test_combined_loss_backward(self) -> None:
        torch.manual_seed(42)

        embeddings = torch.randn(
            8,
            16,
            requires_grad=True,
        )

        logits = torch.randn(
            8,
            4,
            requires_grad=True,
        )

        labels = torch.tensor(
            [0, 0, 1, 1, 2, 2, 3, 3],
            dtype=torch.long,
        )

        criterion = ReIDLoss(
            margin=0.25,
            triplet_weight=1.0,
            classification_weight=0.5,
        )

        loss, metrics = criterion(
            embeddings=embeddings,
            labels=labels,
            logits=logits,
        )

        self.assertTrue(
            torch.isfinite(loss)
        )

        self.assertIn(
            "triplet_loss",
            metrics,
        )

        loss.backward()

        self.assertIsNotNone(
            embeddings.grad
        )

        self.assertIsNotNone(
            logits.grad
        )

        self.assertTrue(
            torch.isfinite(
                embeddings.grad
            ).all()
        )


if __name__ == "__main__":
    unittest.main()