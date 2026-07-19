from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TripletLossResult:
    loss: torch.Tensor
    hardest_positive_distance: torch.Tensor
    hardest_negative_distance: torch.Tensor
    active_fraction: torch.Tensor
    valid_anchor_count: int


def pairwise_cosine_distance(
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Zwraca macierz odległości cosinusowych (B, B).

    Zakres odległości:
        0 -> identyczny kierunek
        1 -> wektory prostopadłe
        2 -> przeciwne kierunki
    """
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings musi mieć kształt (B, D)."
        )

    normalized = F.normalize(
        embeddings,
        p=2,
        dim=1,
        eps=1e-12,
    )

    similarity = normalized @ normalized.T
    similarity = similarity.clamp(-1.0, 1.0)

    distance = 1.0 - similarity

    # Usuwa drobne błędy numeryczne na przekątnej.
    distance.fill_diagonal_(0.0)

    return distance


def batch_hard_cosine_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.25,
) -> TripletLossResult:
    """
    Dla każdego anchora wybiera:

    hardest positive:
        najdalszą próbkę tej samej osoby

    hardest negative:
        najbliższą próbkę innej osoby
    """
    if margin < 0.0:
        raise ValueError(
            "margin nie może być ujemny."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings musi mieć kształt (B, D)."
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels musi mieć kształt (B,)."
        )

    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            "Liczba embeddingów i etykiet musi być taka sama."
        )

    batch_size = embeddings.shape[0]

    if batch_size < 2:
        raise ValueError(
            "Batch musi zawierać co najmniej dwie próbki."
        )

    distances = pairwise_cosine_distance(
        embeddings
    )

    labels = labels.view(-1)

    same_identity = labels[:, None].eq(
        labels[None, :]
    )

    diagonal = torch.eye(
        batch_size,
        dtype=torch.bool,
        device=embeddings.device,
    )

    positive_mask = same_identity & ~diagonal
    negative_mask = ~same_identity

    has_positive = positive_mask.any(dim=1)
    has_negative = negative_mask.any(dim=1)

    valid_anchor_mask = (
        has_positive & has_negative
    )

    valid_anchor_count = int(
        valid_anchor_mask.sum().item()
    )

    if valid_anchor_count == 0:
        raise ValueError(
            "Brak poprawnych anchorów. Batch musi zawierać "
            "co najmniej dwie próbki tej samej osoby oraz "
            "próbki innych osób."
        )

    hardest_positive = distances.masked_fill(
        ~positive_mask,
        -torch.inf,
    ).max(dim=1).values

    hardest_negative = distances.masked_fill(
        ~negative_mask,
        torch.inf,
    ).min(dim=1).values

    hardest_positive = hardest_positive[
        valid_anchor_mask
    ]

    hardest_negative = hardest_negative[
        valid_anchor_mask
    ]

    per_anchor_loss = F.relu(
        hardest_positive
        - hardest_negative
        + margin
    )

    loss = per_anchor_loss.mean()

    active_fraction = (
        per_anchor_loss > 0.0
    ).float().mean()

    return TripletLossResult(
        loss=loss,
        hardest_positive_distance=(
            hardest_positive.mean()
        ),
        hardest_negative_distance=(
            hardest_negative.mean()
        ),
        active_fraction=active_fraction,
        valid_anchor_count=valid_anchor_count,
    )


class ReIDLoss(nn.Module):
    """
    Łączy:

    - Batch-Hard Triplet Loss z odległością cosinusową,
    - Cross Entropy dla pomocniczego klasyfikatora.
    """

    def __init__(
        self,
        margin: float = 0.25,
        triplet_weight: float = 1.0,
        classification_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()

        if triplet_weight < 0.0:
            raise ValueError(
                "triplet_weight nie może być ujemne."
            )

        if classification_weight < 0.0:
            raise ValueError(
                "classification_weight nie może być ujemne."
            )

        self.margin = margin
        self.triplet_weight = triplet_weight
        self.classification_weight = (
            classification_weight
        )
        self.label_smoothing = label_smoothing

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        triplet_result = (
            batch_hard_cosine_triplet_loss(
                embeddings=embeddings,
                labels=labels,
                margin=self.margin,
            )
        )

        classification_loss = embeddings.sum() * 0.0

        if logits is not None:
            if logits.ndim != 2:
                raise ValueError(
                    "logits musi mieć kształt (B, C)."
                )

            if logits.shape[0] != labels.shape[0]:
                raise ValueError(
                    "Logits i labels mają różną liczbę próbek."
                )

            classification_loss = F.cross_entropy(
                logits,
                labels,
                label_smoothing=self.label_smoothing,
            )

        total_loss = (
            self.triplet_weight
            * triplet_result.loss
            + self.classification_weight
            * classification_loss
        )

        metrics = {
            "total_loss": float(
                total_loss.detach().item()
            ),
            "triplet_loss": float(
                triplet_result.loss.detach().item()
            ),
            "classification_loss": float(
                classification_loss.detach().item()
            ),
            "hardest_positive": float(
                triplet_result
                .hardest_positive_distance
                .detach()
                .item()
            ),
            "hardest_negative": float(
                triplet_result
                .hardest_negative_distance
                .detach()
                .item()
            ),
            "active_triplet_fraction": float(
                triplet_result
                .active_fraction
                .detach()
                .item()
            ),
            "valid_anchor_count": float(
                triplet_result.valid_anchor_count
            ),
        }

        return total_loss, metrics