import unittest

import torch

from src.reid.network import SkeletonReIDNetwork


class SkeletonReIDNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

        self.batch_size = 4
        self.sequence_length = 32

        self.keypoints = torch.randn(
            self.batch_size,
            self.sequence_length,
            17,
            3,
        )

        # Confidence musi mieścić się w zakresie 0–1.
        self.keypoints[..., 2] = torch.rand(
            self.batch_size,
            self.sequence_length,
            17,
        )

        self.observed_mask = torch.ones(
            self.batch_size,
            self.sequence_length,
            17,
            dtype=torch.bool,
        )

        self.available_mask = torch.ones(
            self.batch_size,
            self.sequence_length,
            17,
            dtype=torch.bool,
        )

        self.frame_valid_mask = torch.ones(
            self.batch_size,
            self.sequence_length,
            dtype=torch.bool,
        )

    def test_output_shapes(self) -> None:
        model = SkeletonReIDNetwork(
            embedding_dim=128,
            num_classes=20,
        )

        embeddings, logits = model(
            keypoints=self.keypoints,
            observed_mask=self.observed_mask,
            available_mask=self.available_mask,
            frame_valid_mask=self.frame_valid_mask,
        )

        self.assertEqual(
            embeddings.shape,
            (self.batch_size, 128),
        )

        self.assertIsNotNone(logits)

        assert logits is not None

        self.assertEqual(
            logits.shape,
            (self.batch_size, 20),
        )

    def test_embeddings_are_l2_normalized(self) -> None:
        model = SkeletonReIDNetwork(
            embedding_dim=128,
        )

        embeddings, logits = model(
            keypoints=self.keypoints,
            observed_mask=self.observed_mask,
            available_mask=self.available_mask,
            frame_valid_mask=self.frame_valid_mask,
        )

        norms = torch.linalg.vector_norm(
            embeddings,
            dim=1,
        )

        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )

        self.assertIsNone(logits)

    def test_missing_points_do_not_create_nan(self) -> None:
        model = SkeletonReIDNetwork(
            embedding_dim=64,
            num_classes=10,
        )

        self.available_mask[:, 5:10, 8:17] = False
        self.observed_mask[:, 5:10, 8:17] = False
        self.frame_valid_mask[:, 7] = False

        self.keypoints[:, 5:10, 8:17, :2] = 9999.0

        embeddings, logits = model(
            keypoints=self.keypoints,
            observed_mask=self.observed_mask,
            available_mask=self.available_mask,
            frame_valid_mask=self.frame_valid_mask,
        )

        self.assertTrue(
            torch.isfinite(embeddings).all()
        )

        assert logits is not None

        self.assertTrue(
            torch.isfinite(logits).all()
        )

    def test_backward_pass(self) -> None:
        model = SkeletonReIDNetwork(
            embedding_dim=64,
            num_classes=10,
        )

        embeddings, logits = model(
            keypoints=self.keypoints,
            observed_mask=self.observed_mask,
            available_mask=self.available_mask,
            frame_valid_mask=self.frame_valid_mask,
        )

        assert logits is not None

        loss = (
            embeddings[:, 0].mean()
            + logits.square().mean()
        )

        loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        self.assertTrue(
            any(
                gradient is not None
                and torch.isfinite(gradient).all()
                for gradient in gradients
            )
        )


if __name__ == "__main__":
    unittest.main()