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
from src.reid.network import SkeletonReIDNetwork


METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ilids_windows.csv"
)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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

    model = SkeletonReIDNetwork(
        embedding_dim=128,
        frame_feature_dim=128,
        gru_hidden_dim=128,
        gru_layers=1,
        bidirectional=True,
        dropout=0.20,
        num_classes=dataset.num_identities,
    ).to(device)

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

    model.eval()

    with torch.no_grad():
        embeddings, logits = model(
            keypoints=keypoints,
            observed_mask=observed_mask,
            available_mask=available_mask,
            frame_valid_mask=frame_valid_mask,
        )

    norms = torch.linalg.vector_norm(
        embeddings,
        dim=1,
    )

    similarity_matrix = (
        embeddings @ embeddings.T
    )

    print("===== SIEĆ RE-ID =====")
    print(f"Urządzenie:           {device}")
    print(
        f"Liczba parametrów:    "
        f"{count_parameters(model):,}"
    )
    print(
        f"Liczba klas train:    "
        f"{dataset.num_identities}"
    )
    print()

    print("===== WEJŚCIE =====")
    print(
        f"Keypoints:            "
        f"{tuple(keypoints.shape)}"
    )
    print()

    print("===== WYJŚCIE =====")
    print(
        f"Embedding:            "
        f"{tuple(embeddings.shape)}"
    )

    if logits is not None:
        print(
            f"Logits:               "
            f"{tuple(logits.shape)}"
        )

    print(
        f"Macierz podobieństwa: "
        f"{tuple(similarity_matrix.shape)}"
    )

    print()
    print(
        f"Norma embeddingu min: "
        f"{norms.min().item():.6f}"
    )

    print(
        f"Norma embeddingu avg: "
        f"{norms.mean().item():.6f}"
    )

    print(
        f"Norma embeddingu max: "
        f"{norms.max().item():.6f}"
    )

    print()
    print(
        f"Person IDs:           "
        f"{batch['person_id'].tolist()}"
    )


if __name__ == "__main__":
    main()