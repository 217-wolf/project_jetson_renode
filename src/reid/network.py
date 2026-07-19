from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SkeletonReIDNetwork(nn.Module):
    """
    Sieć Re-ID analizująca sekwencję znormalizowanych szkieletów.

    Wejście:
        keypoints:         (B, T, 17, 3)
        observed_mask:     (B, T, 17)
        available_mask:    (B, T, 17)
        frame_valid_mask:  (B, T)

    Wyjście:
        embedding:         (B, embedding_dim)
        logits:            (B, num_classes) albo None
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        embedding_dim: int = 128,
        frame_feature_dim: int = 128,
        gru_hidden_dim: int = 128,
        gru_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.20,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()

        if num_keypoints <= 0:
            raise ValueError(
                "num_keypoints musi być większe od zera."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim musi być większe od zera."
            )

        if gru_layers <= 0:
            raise ValueError(
                "gru_layers musi być większe od zera."
            )

        self.num_keypoints = num_keypoints
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Dla każdego punktu:
        # x, y, confidence, observed, available
        features_per_keypoint = 5

        # Dodatkowo:
        # frame_valid oraz średnia dostępność punktów w klatce
        frame_input_dim = (
            num_keypoints * features_per_keypoint
            + 2
        )

        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(frame_input_dim),
            nn.Linear(frame_input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, frame_feature_dim),
            nn.GELU(),
        )

        gru_dropout = dropout if gru_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=frame_feature_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )

        gru_output_dim = gru_hidden_dim * (
            2 if bidirectional else 1
        )

        self.temporal_norm = nn.LayerNorm(
            gru_output_dim
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(gru_output_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

        if num_classes is None:
            self.classifier = None
        else:
            if num_classes <= 1:
                raise ValueError(
                    "num_classes musi być większe od 1."
                )

            self.classifier = nn.Linear(
                embedding_dim,
                num_classes,
            )

    def _validate_inputs(
        self,
        keypoints: torch.Tensor,
        observed_mask: torch.Tensor,
        available_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> None:
        if keypoints.ndim != 4:
            raise ValueError(
                "keypoints musi mieć kształt "
                "(B, T, 17, 3)."
            )

        batch_size, sequence_length, point_count, feature_count = (
            keypoints.shape
        )

        if point_count != self.num_keypoints:
            raise ValueError(
                f"Oczekiwano {self.num_keypoints} punktów, "
                f"otrzymano {point_count}."
            )

        if feature_count != 3:
            raise ValueError(
                "Ostatni wymiar keypoints musi wynosić 3."
            )

        expected_point_mask_shape = (
            batch_size,
            sequence_length,
            self.num_keypoints,
        )

        if observed_mask.shape != expected_point_mask_shape:
            raise ValueError(
                "Niepoprawny kształt observed_mask: "
                f"{tuple(observed_mask.shape)}."
            )

        if available_mask.shape != expected_point_mask_shape:
            raise ValueError(
                "Niepoprawny kształt available_mask: "
                f"{tuple(available_mask.shape)}."
            )

        expected_frame_mask_shape = (
            batch_size,
            sequence_length,
        )

        if frame_valid_mask.shape != expected_frame_mask_shape:
            raise ValueError(
                "Niepoprawny kształt frame_valid_mask: "
                f"{tuple(frame_valid_mask.shape)}."
            )

    def _prepare_frame_features(
        self,
        keypoints: torch.Tensor,
        observed_mask: torch.Tensor,
        available_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keypoints = keypoints.float()

        observed = observed_mask.bool()
        available = available_mask.bool()
        frame_valid = frame_valid_mask.bool()

        # Niedostępne współrzędne nie mogą wpływać na model.
        xy = torch.where(
            available.unsqueeze(-1),
            keypoints[..., :2],
            torch.zeros_like(keypoints[..., :2]),
        )

        # Confidence zachowujemy tylko dla punktów
        # rzeczywiście wykrytych przez YOLO.
        confidence = torch.where(
            observed,
            keypoints[..., 2],
            torch.zeros_like(keypoints[..., 2]),
        ).unsqueeze(-1)

        point_features = torch.cat(
            [
                xy,
                confidence,
                observed.float().unsqueeze(-1),
                available.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        batch_size, sequence_length = keypoints.shape[:2]

        frame_features = point_features.reshape(
            batch_size,
            sequence_length,
            -1,
        )

        availability_ratio = (
            available.float()
            .mean(dim=-1, keepdim=True)
        )

        frame_features = torch.cat(
            [
                frame_features,
                frame_valid.float().unsqueeze(-1),
                availability_ratio,
            ],
            dim=-1,
        )

        return frame_features, availability_ratio.squeeze(-1)

    def forward(
        self,
        keypoints: torch.Tensor,
        observed_mask: torch.Tensor,
        available_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_inputs(
            keypoints=keypoints,
            observed_mask=observed_mask,
            available_mask=available_mask,
            frame_valid_mask=frame_valid_mask,
        )

        (
            frame_features,
            availability_ratio,
        ) = self._prepare_frame_features(
            keypoints=keypoints,
            observed_mask=observed_mask,
            available_mask=available_mask,
            frame_valid_mask=frame_valid_mask,
        )

        encoded_frames = self.frame_encoder(
            frame_features
        )

        temporal_features, _ = self.gru(
            encoded_frames
        )

        temporal_features = self.temporal_norm(
            temporal_features
        )

        # Klatki dobre dostają większą wagę.
        # Klatki częściowo uzupełnione nadal mogą wnosić informację.
        frame_weights = (
            0.75 * availability_ratio
            + 0.25 * frame_valid_mask.float()
        )

        weight_sum = frame_weights.sum(
            dim=1,
            keepdim=True,
        )

        # Zabezpieczenie, gdyby cały bufor był pusty.
        fallback_weights = torch.ones_like(
            frame_weights
        )

        frame_weights = torch.where(
            weight_sum > 0,
            frame_weights,
            fallback_weights,
        )

        weight_sum = frame_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)

        pooled_features = (
            temporal_features
            * frame_weights.unsqueeze(-1)
        ).sum(dim=1) / weight_sum

        raw_embedding = self.embedding_head(
            pooled_features
        )

        embedding = F.normalize(
            raw_embedding,
            p=2,
            dim=1,
            eps=1e-12,
        )

        logits: torch.Tensor | None = None

        if self.classifier is not None:
            logits = self.classifier(
                raw_embedding
            )

        return embedding, logits