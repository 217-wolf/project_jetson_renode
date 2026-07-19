from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets.identity_sampler import IdentityBatchSampler
from src.datasets.skeleton_dataset import SkeletonWindowDataset
from src.reid.evaluation import (
    collect_embeddings,
    evaluate_cross_camera_retrieval,
)
from src.reid.losses import ReIDLoss
from src.reid.network import SkeletonReIDNetwork


METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ilids_windows.csv"
)

MODELS_DIRECTORY = PROJECT_ROOT / "models"
RUNS_DIRECTORY = PROJECT_ROOT / "runs"

BEST_MODEL_PATH = (
    MODELS_DIRECTORY
    / "skeleton_reid_best.pt"
)

LAST_MODEL_PATH = (
    MODELS_DIRECTORY
    / "skeleton_reid_last.pt"
)

HISTORY_PATH = (
    RUNS_DIRECTORY
    / "reid_training_history.csv"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trening sieci Skeleton Re-ID."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "train.yaml",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint, z którego wznowić trening.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Nadpisuje liczbę epok z pliku YAML.",
    )

    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=None,
        help=(
            "Nadpisuje liczbę batchy na epokę. "
            "Przydatne do szybkiego testu."
        ),
    )

    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono konfiguracji: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Plik YAML nie zawiera poprawnej konfiguracji."
        )

    return config


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "keypoints": batch["keypoints"].to(
            device,
            non_blocking=True,
        ),
        "observed_mask": batch["observed_mask"].to(
            device,
            non_blocking=True,
        ),
        "available_mask": batch["available_mask"].to(
            device,
            non_blocking=True,
        ),
        "frame_valid_mask": batch["frame_valid_mask"].to(
            device,
            non_blocking=True,
        ),
        "labels": batch["label"].to(
            device,
            non_blocking=True,
        ),
    }


def train_one_epoch(
    model: SkeletonReIDNetwork,
    data_loader: DataLoader,
    sampler: IdentityBatchSampler,
    criterion: ReIDLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_gradient_norm: float,
) -> dict[str, float]:
    model.train()
    sampler.set_epoch(epoch)

    metric_sums = {
        "total_loss": 0.0,
        "triplet_loss": 0.0,
        "classification_loss": 0.0,
        "hardest_positive": 0.0,
        "hardest_negative": 0.0,
        "active_triplet_fraction": 0.0,
    }

    batch_count = 0

    for batch_index, batch in enumerate(
        data_loader,
        start=1,
    ):
        tensors = move_batch_to_device(
            batch=batch,
            device=device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        embeddings, logits = model(
            keypoints=tensors["keypoints"],
            observed_mask=tensors["observed_mask"],
            available_mask=tensors["available_mask"],
            frame_valid_mask=tensors["frame_valid_mask"],
        )

        loss, metrics = criterion(
            embeddings=embeddings,
            labels=tensors["labels"],
            logits=logits,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Nieskończona strata w batchu {batch_index}."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_gradient_norm,
        )

        optimizer.step()

        for metric_name in metric_sums:
            metric_sums[metric_name] += metrics[
                metric_name
            ]

        batch_count += 1

        if (
            batch_index % 20 == 0
            or batch_index == len(data_loader)
        ):
            print(
                f"    batch "
                f"{batch_index:03d}/{len(data_loader):03d} | "
                f"loss={metrics['total_loss']:.4f} | "
                f"triplet={metrics['triplet_loss']:.4f} | "
                f"ce={metrics['classification_loss']:.4f}"
            )

    if batch_count == 0:
        raise RuntimeError(
            "Epoka nie zawierała żadnych batchy."
        )

    return {
        metric_name: metric_sum / batch_count
        for metric_name, metric_sum
        in metric_sums.items()
    }


def save_checkpoint(
    path: Path,
    model: SkeletonReIDNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_rank1: float,
    best_epoch: int,
    config: dict[str, Any],
    train_dataset: SkeletonWindowDataset,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "best_rank1": best_rank1,
        "best_epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "person_id_to_label": (
            train_dataset.person_id_to_label
        ),
        "label_to_person_id": (
            train_dataset.label_to_person_id
        ),
        "num_train_identities": (
            train_dataset.num_identities
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


def append_history(
    row: dict[str, object],
    reset_file: bool,
) -> None:
    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(row)

    mode = "w" if reset_file else "a"

    with HISTORY_PATH.open(
        mode,
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if reset_file:
            writer.writeheader()

        writer.writerow(row)


def main() -> None:
    arguments = parse_arguments()
    config = load_config(
        arguments.config
    )

    if arguments.epochs is not None:
        config["epochs"] = arguments.epochs

    if arguments.batches_per_epoch is not None:
        config["batches_per_epoch"] = (
            arguments.batches_per_epoch
        )

    seed = int(config["seed"])
    set_random_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision(
            "high"
        )

    MODELS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUNS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset = SkeletonWindowDataset(
        metadata_path=METADATA_PATH,
        split="train",
        project_root=PROJECT_ROOT,
    )

    val_dataset = SkeletonWindowDataset(
        metadata_path=METADATA_PATH,
        split="val",
        project_root=PROJECT_ROOT,
    )

    train_sampler = IdentityBatchSampler(
        dataset=train_dataset,
        identities_per_batch=int(
            config["identities_per_batch"]
        ),
        samples_per_identity=int(
            config["samples_per_identity"]
        ),
        batches_per_epoch=config[
            "batches_per_epoch"
        ],
        minimum_samples_per_identity=int(
            config["minimum_samples_per_identity"]
        ),
        seed=seed,
    )

    num_workers = int(
        config["num_workers"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(
            config["validation_batch_size"]
        ),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model_config = {
        "embedding_dim": int(
            config["embedding_dim"]
        ),
        "frame_feature_dim": int(
            config["frame_feature_dim"]
        ),
        "gru_hidden_dim": int(
            config["gru_hidden_dim"]
        ),
        "gru_layers": int(
            config["gru_layers"]
        ),
        "bidirectional": bool(
            config["bidirectional"]
        ),
        "dropout": float(
            config["dropout"]
        ),
        "num_classes": (
            train_dataset.num_identities
        ),
    }

    model = SkeletonReIDNetwork(
        **model_config
    ).to(device)

    criterion = ReIDLoss(
        margin=float(
            config["triplet_margin"]
        ),
        triplet_weight=float(
            config["triplet_weight"]
        ),
        classification_weight=float(
            config["classification_weight"]
        ),
        label_smoothing=float(
            config["label_smoothing"]
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            config["learning_rate"]
        ),
        weight_decay=float(
            config["weight_decay"]
        ),
    )

    epochs = int(config["epochs"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
    )

    start_epoch = 1
    best_rank1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    if arguments.resume is not None:
        if not arguments.resume.exists():
            raise FileNotFoundError(
                f"Nie znaleziono checkpointu: "
                f"{arguments.resume}"
            )

        checkpoint = torch.load(
            arguments.resume,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch = int(
            checkpoint["epoch"]
        ) + 1

        best_rank1 = float(
            checkpoint.get(
                "best_rank1",
                -1.0,
            )
        )

        best_epoch = int(
            checkpoint.get(
                "best_epoch",
                0,
            )
        )

        print(
            f"Wznowiono trening od epoki "
            f"{start_epoch}."
        )

    print("===== TRENING SKELETON RE-ID =====")
    print(f"Urządzenie:             {device}")
    print(
        f"Bufory train:           "
        f"{len(train_dataset)}"
    )
    print(
        f"Osoby train:            "
        f"{train_dataset.num_identities}"
    )
    print(
        f"Bufory val:             "
        f"{len(val_dataset)}"
    )
    print(
        f"Batch:                  "
        f"{train_sampler.batch_size}"
    )
    print(
        f"Batche na epokę:        "
        f"{len(train_sampler)}"
    )
    print(f"Epoki:                  {epochs}")
    print()

    reset_history = (
        arguments.resume is None
    )

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):
        current_learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"===== EPOKA {epoch}/{epochs} ====="
        )

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            sampler=train_sampler,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_gradient_norm=float(
                config["max_gradient_norm"]
            ),
        )

        (
            val_embeddings,
            val_person_ids,
            val_camera_ids,
        ) = collect_embeddings(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        retrieval_metrics = (
            evaluate_cross_camera_retrieval(
                embeddings=val_embeddings,
                person_ids=val_person_ids,
                camera_ids=val_camera_ids,
            )
        )

        scheduler.step()

        print()
        print(
            f"Train loss:             "
            f"{train_metrics['total_loss']:.6f}"
        )

        print(
            f"Train triplet:          "
            f"{train_metrics['triplet_loss']:.6f}"
        )

        print(
            f"Train classification:   "
            f"{train_metrics['classification_loss']:.6f}"
        )

        print(
            f"Hardest positive:       "
            f"{train_metrics['hardest_positive']:.6f}"
        )

        print(
            f"Hardest negative:       "
            f"{train_metrics['hardest_negative']:.6f}"
        )

        print(
            f"Aktywne triplety:       "
            f"{100.0 * train_metrics['active_triplet_fraction']:.2f}%"
        )

        print()
        print(
            f"Val Rank-1:             "
            f"{100.0 * retrieval_metrics.rank1:.2f}%"
        )

        print(
            f"Val positive distance:  "
            f"{retrieval_metrics.mean_positive_distance:.6f}"
        )

        print(
            f"Val negative distance:  "
            f"{retrieval_metrics.mean_negative_distance:.6f}"
        )

        print(
            f"Val queries:            "
            f"{retrieval_metrics.query_count}"
        )

        improved = (
            retrieval_metrics.rank1
            > best_rank1
        )

        if improved:
            best_rank1 = (
                retrieval_metrics.rank1
            )

            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            path=LAST_MODEL_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_rank1=best_rank1,
            best_epoch=best_epoch,
            config={
                **config,
                "model": model_config,
            },
            train_dataset=train_dataset,
        )

        if improved:
            save_checkpoint(
                path=BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_rank1=best_rank1,
                best_epoch=best_epoch,
                config={
                    **config,
                    "model": model_config,
                },
                train_dataset=train_dataset,
            )

            print(
                f"Zapisano nowy najlepszy model: "
                f"Rank-1={100.0 * best_rank1:.2f}%"
            )

        history_row = {
            "epoch": epoch,
            "learning_rate": (
                f"{current_learning_rate:.10f}"
            ),
            "train_total_loss": (
                f"{train_metrics['total_loss']:.8f}"
            ),
            "train_triplet_loss": (
                f"{train_metrics['triplet_loss']:.8f}"
            ),
            "train_classification_loss": (
                f"{train_metrics['classification_loss']:.8f}"
            ),
            "train_hardest_positive": (
                f"{train_metrics['hardest_positive']:.8f}"
            ),
            "train_hardest_negative": (
                f"{train_metrics['hardest_negative']:.8f}"
            ),
            "train_active_triplet_fraction": (
                f"{train_metrics['active_triplet_fraction']:.8f}"
            ),
            "val_rank1": (
                f"{retrieval_metrics.rank1:.8f}"
            ),
            "val_positive_distance": (
                f"{retrieval_metrics.mean_positive_distance:.8f}"
            ),
            "val_negative_distance": (
                f"{retrieval_metrics.mean_negative_distance:.8f}"
            ),
        }

        append_history(
            row=history_row,
            reset_file=reset_history,
        )

        reset_history = False

        print(
            f"Najlepszy Rank-1:       "
            f"{100.0 * best_rank1:.2f}% "
            f"(epoka {best_epoch})"
        )

        print()

        patience = int(
            config["early_stopping_patience"]
        )

        if (
            patience > 0
            and epochs_without_improvement
            >= patience
        ):
            print(
                "Zatrzymanie treningu: brak poprawy "
                f"przez {patience} epok."
            )
            break

    print("===== KONIEC TRENINGU =====")
    print(
        f"Najlepszy Rank-1: "
        f"{100.0 * best_rank1:.2f}%"
    )
    print(f"Najlepsza epoka: {best_epoch}")
    print(f"Najlepszy model: {BEST_MODEL_PATH}")
    print(f"Ostatni model:   {LAST_MODEL_PATH}")
    print(f"Historia:        {HISTORY_PATH}")


if __name__ == "__main__":
    main()