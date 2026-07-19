from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class RetrievalMetrics:
    rank1: float
    mean_positive_distance: float
    mean_negative_distance: float
    query_count: int
    prototype_count: int


@torch.no_grad()
def collect_embeddings(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Oblicza embeddingi wszystkich buforów z DataLoadera.
    """
    model.eval()

    all_embeddings: list[torch.Tensor] = []
    all_person_ids: list[torch.Tensor] = []
    all_camera_ids: list[torch.Tensor] = []

    for batch in data_loader:
        keypoints = batch["keypoints"].to(
            device,
            non_blocking=True,
        )

        observed_mask = batch["observed_mask"].to(
            device,
            non_blocking=True,
        )

        available_mask = batch["available_mask"].to(
            device,
            non_blocking=True,
        )

        frame_valid_mask = batch["frame_valid_mask"].to(
            device,
            non_blocking=True,
        )

        embeddings, _ = model(
            keypoints=keypoints,
            observed_mask=observed_mask,
            available_mask=available_mask,
            frame_valid_mask=frame_valid_mask,
        )

        all_embeddings.append(
            embeddings.detach().cpu()
        )

        all_person_ids.append(
            batch["person_id"].detach().cpu()
        )

        all_camera_ids.append(
            batch["camera_id"].detach().cpu()
        )

    if not all_embeddings:
        raise RuntimeError(
            "DataLoader walidacyjny nie zwrócił żadnych próbek."
        )

    return (
        torch.cat(all_embeddings, dim=0),
        torch.cat(all_person_ids, dim=0),
        torch.cat(all_camera_ids, dim=0),
    )


def build_camera_prototypes(
    embeddings: torch.Tensor,
    person_ids: torch.Tensor,
    camera_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Łączy wszystkie okna tej samej osoby z tej samej kamery
    w jeden uśredniony i znormalizowany prototyp.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings musi mieć kształt (N, D)."
        )

    if person_ids.shape != (embeddings.shape[0],):
        raise ValueError(
            "Niepoprawny kształt person_ids."
        )

    if camera_ids.shape != (embeddings.shape[0],):
        raise ValueError(
            "Niepoprawny kształt camera_ids."
        )

    grouped_indices: dict[
        tuple[int, int],
        list[int],
    ] = {}

    for index, (person_id, camera_id) in enumerate(
        zip(person_ids.tolist(), camera_ids.tolist())
    ):
        key = (int(person_id), int(camera_id))

        grouped_indices.setdefault(
            key,
            [],
        ).append(index)

    prototype_embeddings: list[torch.Tensor] = []
    prototype_person_ids: list[int] = []
    prototype_camera_ids: list[int] = []

    for (
        person_id,
        camera_id,
    ), indices in sorted(grouped_indices.items()):
        selected = embeddings[indices]

        prototype = selected.mean(
            dim=0,
            keepdim=True,
        )

        prototype = F.normalize(
            prototype,
            p=2,
            dim=1,
            eps=1e-12,
        ).squeeze(0)

        prototype_embeddings.append(prototype)
        prototype_person_ids.append(person_id)
        prototype_camera_ids.append(camera_id)

    return (
        torch.stack(prototype_embeddings),
        torch.tensor(
            prototype_person_ids,
            dtype=torch.long,
        ),
        torch.tensor(
            prototype_camera_ids,
            dtype=torch.long,
        ),
    )


def evaluate_cross_camera_retrieval(
    embeddings: torch.Tensor,
    person_ids: torch.Tensor,
    camera_ids: torch.Tensor,
) -> RetrievalMetrics:
    """
    Sprawdza dopasowanie:

        cam1 -> cam2
        cam2 -> cam1

    Rank-1 oznacza procent zapytań, dla których najbliższy
    prototyp z drugiej kamery należy do tej samej osoby.
    """
    (
        prototypes,
        prototype_person_ids,
        prototype_camera_ids,
    ) = build_camera_prototypes(
        embeddings=embeddings,
        person_ids=person_ids,
        camera_ids=camera_ids,
    )

    correct = 0
    query_count = 0

    positive_distances: list[float] = []
    negative_distances: list[float] = []

    for query_camera, gallery_camera in (
        (1, 2),
        (2, 1),
    ):
        query_indices = torch.nonzero(
            prototype_camera_ids == query_camera,
            as_tuple=False,
        ).flatten()

        gallery_indices = torch.nonzero(
            prototype_camera_ids == gallery_camera,
            as_tuple=False,
        ).flatten()

        if len(query_indices) == 0 or len(gallery_indices) == 0:
            continue

        gallery_embeddings = prototypes[
            gallery_indices
        ]

        gallery_person_ids = prototype_person_ids[
            gallery_indices
        ]

        for query_index in query_indices.tolist():
            query_person_id = int(
                prototype_person_ids[query_index]
            )

            positive_mask = (
                gallery_person_ids
                == query_person_id
            )

            # Pomijamy osobę, gdy nie ma prototypu z drugiej kamery.
            if not bool(positive_mask.any()):
                continue

            query_embedding = prototypes[
                query_index
            ]

            similarities = (
                gallery_embeddings
                @ query_embedding
            )

            distances = 1.0 - similarities

            nearest_index = int(
                torch.argmin(distances).item()
            )

            predicted_person_id = int(
                gallery_person_ids[
                    nearest_index
                ].item()
            )

            correct += int(
                predicted_person_id
                == query_person_id
            )

            query_count += 1

            positive_distances.extend(
                distances[
                    positive_mask
                ].tolist()
            )

            negative_mask = ~positive_mask

            negative_distances.extend(
                distances[
                    negative_mask
                ].tolist()
            )

    if query_count == 0:
        raise RuntimeError(
            "Brak osób obecnych w obu kamerach w walidacji."
        )

    mean_positive = (
        sum(positive_distances)
        / len(positive_distances)
    )

    mean_negative = (
        sum(negative_distances)
        / len(negative_distances)
    )

    return RetrievalMetrics(
        rank1=correct / query_count,
        mean_positive_distance=mean_positive,
        mean_negative_distance=mean_negative,
        query_count=query_count,
        prototype_count=len(prototypes),
    )