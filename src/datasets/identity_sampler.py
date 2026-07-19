from __future__ import annotations

import math
import random
from collections.abc import Iterator

from torch.utils.data import Sampler

from src.datasets.skeleton_dataset import (
    SkeletonWindowDataset,
)


class IdentityBatchSampler(Sampler):
    """
    Tworzy paczki w układzie:

        P różnych osób × K próbek każdej osoby

    Przykład:
        8 osób × 4 próbki = batch 32

    Wymagane są minimum dwie różne próbki danej osoby,
    aby Triplet Loss miał prawdziwą parę positive.
    """

    def __init__(
        self,
        dataset: SkeletonWindowDataset,
        identities_per_batch: int = 8,
        samples_per_identity: int = 4,
        batches_per_epoch: int | None = None,
        minimum_samples_per_identity: int = 2,
        seed: int = 42,
    ) -> None:
        if identities_per_batch < 2:
            raise ValueError(
                "identities_per_batch musi wynosić "
                "co najmniej 2."
            )

        if samples_per_identity < 2:
            raise ValueError(
                "samples_per_identity musi wynosić "
                "co najmniej 2."
            )

        if minimum_samples_per_identity < 2:
            raise ValueError(
                "minimum_samples_per_identity musi "
                "wynosić co najmniej 2."
            )

        self.dataset = dataset
        self.identities_per_batch = (
            identities_per_batch
        )
        self.samples_per_identity = (
            samples_per_identity
        )
        self.seed = seed
        self.epoch = 0

        self.identity_to_indices = {
            person_id: indices
            for person_id, indices
            in dataset.identity_to_indices.items()
            if len(indices)
            >= minimum_samples_per_identity
        }

        self.identities = sorted(
            self.identity_to_indices
        )

        if (
            len(self.identities)
            < self.identities_per_batch
        ):
            raise RuntimeError(
                "Za mało osób z dostateczną liczbą "
                "próbek. Dostępne osoby: "
                f"{len(self.identities)}, wymagane: "
                f"{self.identities_per_batch}."
            )

        self.batch_size = (
            self.identities_per_batch
            * self.samples_per_identity
        )

        eligible_sample_count = sum(
            len(indices)
            for indices
            in self.identity_to_indices.values()
        )

        if batches_per_epoch is None:
            self.batches_per_epoch = max(
                1,
                math.ceil(
                    eligible_sample_count
                    / self.batch_size
                ),
            )
        else:
            if batches_per_epoch <= 0:
                raise ValueError(
                    "batches_per_epoch musi być "
                    "większe od zera."
                )

            self.batches_per_epoch = (
                batches_per_epoch
            )

    def set_epoch(self, epoch: int) -> None:
        """
        Pozwala otrzymać inne losowanie w każdej epoce,
        ale zachować pełną powtarzalność treningu.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        random_generator = random.Random(
            self.seed + self.epoch
        )

        for _ in range(self.batches_per_epoch):
            selected_identities = (
                random_generator.sample(
                    self.identities,
                    self.identities_per_batch,
                )
            )

            batch_indices: list[int] = []

            for person_id in selected_identities:
                available_indices = (
                    self.identity_to_indices[
                        person_id
                    ]
                )

                if (
                    len(available_indices)
                    >= self.samples_per_identity
                ):
                    selected_indices = (
                        random_generator.sample(
                            available_indices,
                            self.samples_per_identity,
                        )
                    )
                else:
                    # Najpierw bierzemy wszystkie różne
                    # próbki, a brakujące uzupełniamy
                    # losowaniem z powtórzeniami.
                    selected_indices = (
                        available_indices.copy()
                    )

                    random_generator.shuffle(
                        selected_indices
                    )

                    while (
                        len(selected_indices)
                        < self.samples_per_identity
                    ):
                        selected_indices.append(
                            random_generator.choice(
                                available_indices
                            )
                        )

                batch_indices.extend(
                    selected_indices
                )

            random_generator.shuffle(
                batch_indices
            )

            yield batch_indices