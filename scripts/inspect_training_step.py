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
from src.reid.losses import ReIDLoss
from src.reid.network import SkeletonReIDNetwork


METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ilids_windows.csv"
)


def calculate_gradient_norm(
    model: torch.nn.Module,
) -> float:
    squared_norm = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        gradient_norm = (
            parameter.grad.detach().norm(2).item()
        )

        squared_norm += gradient_norm ** 2

    return squared_norm ** 0.5


def main() -> None:
    torch.manual_seed(42)

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

    criterion = ReIDLoss(
        margin=0.25,
        triplet_weight=1.0,
        classification_weight=0.5,
        label_smoothing=0.1,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

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

    frame_valid_mask = batch[
        "frame_valid_mask"
    ].to(
        device,
        non_blocking=True,
    )

    labels = batch["label"].to(
        device,
        non_blocking=True,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    embeddings, logits = model(
        keypoints=keypoints,
        observed_mask=observed_mask,
        available_mask=available_mask,
        frame_valid_mask=frame_valid_mask,
    )

    loss, metrics = criterion(
        embeddings=embeddings,
        labels=labels,
        logits=logits,
    )

    loss.backward()

    gradient_norm_before_clip = (
        calculate_gradient_norm(model)
    )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0,
    )

    optimizer.step()

    print("===== PRÓBNY KROK TRENINGOWY =====")
    print(f"Urządzenie:               {device}")
    print(f"Batch:                    {len(labels)}")
    print(
        f"Poprawne anchory:         "
        f"{int(metrics['valid_anchor_count'])}"
    )
    print()

    print(
        f"Łączna strata:            "
        f"{metrics['total_loss']:.6f}"
    )

    print(
        f"Triplet loss:             "
        f"{metrics['triplet_loss']:.6f}"
    )

    print(
        f"Classification loss:      "
        f"{metrics['classification_loss']:.6f}"
    )

    print()
    print(
        f"Hardest positive:         "
        f"{metrics['hardest_positive']:.6f}"
    )

    print(
        f"Hardest negative:         "
        f"{metrics['hardest_negative']:.6f}"
    )

    print(
        f"Aktywne triplety:         "
        f"{100.0 * metrics['active_triplet_fraction']:.2f}%"
    )

    print()
    print(
        f"Norma gradientu:          "
        f"{gradient_norm_before_clip:.6f}"
    )

    print()
    print("Krok optimizer.step() wykonany poprawnie.")


if __name__ == "__main__":
    main()