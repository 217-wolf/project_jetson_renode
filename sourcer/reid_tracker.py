"""Śledzenie obiektów ReID oraz uporządkowany zapis jego stanu na dysku."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _safe_component(value: str) -> str:
    """Zamień nazwę klasy na bezpieczną nazwę katalogu."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "unknown"


class EmbeddingStore:
    """Zapisuje stan ReID bez luźnych plików w ``patterns_database``.

    Każda tożsamość ma własny katalog z prototypem, metadanymi i galerią
    reprezentatywnych ujęć. Katalog ``reid`` jest niezależny od katalogu
    ``known`` obsługiwanego przez :class:`PatternsDatabase`.
    """

    def __init__(self, database_path: Path | str = "patterns_database"):
        self.database_path = Path(database_path)
        self.reid_root = self.database_path / "reid"
        self.reid_root.mkdir(parents=True, exist_ok=True)

    def _identity_dir(self, class_name: str, identity_id: int) -> Path:
        return self.reid_root / _safe_component(class_name) / f"identity_{identity_id:06d}"

    def save(self, identity_id: int, identity: Dict) -> Path:
        """Zsynchronizuj prototyp, galerię i metadane jednej tożsamości."""
        class_name = identity["class"]
        identity_dir = self._identity_dir(class_name, identity_id)
        gallery_dir = identity_dir / "gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)

        np.save(identity_dir / "prototype.npy", np.asarray(identity["prototype"], dtype=np.float32))

        expected_gallery_files = set()
        for index, embedding in enumerate(identity.get("gallery", [])):
            gallery_file = gallery_dir / f"view_{index:03d}.npy"
            np.save(gallery_file, np.asarray(embedding, dtype=np.float32))
            expected_gallery_files.add(gallery_file)

        # Galeria ma stały limit; usuń tylko stare widoki tej tożsamości.
        for old_file in gallery_dir.glob("view_*.npy"):
            if old_file not in expected_gallery_files:
                old_file.unlink()

        metadata = {
            "schema_version": 1,
            "identity_id": identity_id,
            "class": class_name,
            "confidence": float(identity.get("confidence", 0.0)),
            "samples": int(identity.get("samples", 0)),
            "last_seen": identity.get("last_seen", ""),
            "gallery_size": len(identity.get("gallery", [])),
        }
        with (identity_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        return identity_dir

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        """Zwróć liczbę tożsamości i widoków galerii dla każdej klasy."""
        summary: Dict[str, Dict[str, int]] = {}
        if not self.reid_root.exists():
            return summary

        for class_dir in self.reid_root.iterdir():
            if not class_dir.is_dir():
                continue
            identity_dirs = [item for item in class_dir.iterdir() if item.is_dir()]
            gallery_samples = sum(
                len(list((identity_dir / "gallery").glob("view_*.npy")))
                for identity_dir in identity_dirs
            )
            summary[class_dir.name] = {
                "identities": len(identity_dirs),
                "gallery_samples": gallery_samples,
            }
        return summary


class PersistentReIDTracker:
    """Tracker ReID z pamięcią wieloperspektywiczną i opcjonalnym zapisem stanu."""

    def __init__(
        self,
        similarity_threshold: float = 0.58,
        bbox_overlap_threshold: float = 0.15,
        max_missing: int = 150,
        max_gallery_size: int = 5,
        store: Optional[EmbeddingStore] = None,
        persist_every_frames: int = 30,
    ):
        self.similarity_threshold = similarity_threshold
        self.bbox_overlap_threshold = bbox_overlap_threshold
        self.max_missing = max_missing
        self.max_gallery_size = max_gallery_size
        self.store = store
        self.persist_every_frames = max(1, persist_every_frames)
        self.frame_counter = 0
        self.identities: Dict[int, Dict] = {}
        self.next_id = 1

    def reset(self) -> None:
        """Wyczyść pamięć bieżącej sesji; zapisane dane ReID pozostają na dysku."""
        self.identities.clear()
        self.next_id = 1
        self.frame_counter = 0
        logger.info("Zresetowano pamięć trackera ReID dla bieżącej sesji.")

    @staticmethod
    def _normalise(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    @staticmethod
    def _similarity(first: np.ndarray, second: np.ndarray) -> float:
        first = np.asarray(first, dtype=np.float32).ravel()
        second = np.asarray(second, dtype=np.float32).ravel()
        first_norm = np.linalg.norm(first)
        second_norm = np.linalg.norm(second)
        if first_norm == 0 or second_norm == 0 or first.shape != second.shape:
            return -1.0
        return float(np.dot(first, second) / (first_norm * second_norm))

    @staticmethod
    def _bbox_overlap(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def _similarity_to_identity(self, embedding: np.ndarray, identity: Dict) -> float:
        scores = [self._similarity(embedding, identity["prototype"])]
        scores.extend(self._similarity(embedding, gallery_emb) for gallery_emb in identity.get("gallery", []))
        return max(scores)

    def _candidate_score(self, identity: Dict, detection: Dict) -> float:
        if identity["class"] != detection["class"]:
            return -1.0

        similarity = self._similarity_to_identity(detection["embedding"], identity)
        overlap = self._bbox_overlap(detection["bbox"], identity["bbox"])
        missing = identity.get("missing", 0)

        if missing <= 30 and overlap >= self.bbox_overlap_threshold:
            combined = 0.4 * overlap + 0.6 * max(0.0, similarity)
            if similarity >= 0.40 or combined >= 0.40:
                return combined

        if missing <= self.max_missing:
            if similarity >= self.similarity_threshold:
                return similarity
            if missing <= 45 and similarity >= self.similarity_threshold - 0.06:
                return similarity

        return similarity if similarity >= self.similarity_threshold + 0.06 else -1.0

    def _persist_identity(self, identity_id: int, *, force: bool = False) -> Optional[str]:
        if self.store is None:
            return None
        if not force and self.frame_counter % self.persist_every_frames != 0:
            return self.identities[identity_id].get("embedding_path")

        identity_dir = self.store.save(identity_id, self.identities[identity_id])
        prototype_path = str(identity_dir / "prototype.npy")
        self.identities[identity_id]["embedding_path"] = prototype_path
        return prototype_path

    def flush(self) -> None:
        """Zapisz końcowy stan sesji, np. przy zatrzymaniu kolektora."""
        for identity_id in self.identities:
            self._persist_identity(identity_id, force=True)

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Przypisz stabilne ID detekcjom i okresowo zapisz ich stan."""
        self.frame_counter += 1
        available_ids = set(self.identities)
        assigned_ids = set()
        assignments = []

        for detection in detections:
            best_id = None
            best_score = -1.0
            for identity_id in available_ids:
                score = self._candidate_score(self.identities[identity_id], detection)
                if score > best_score:
                    best_id, best_score = identity_id, score

            confidence = float(detection.get("confidence", 0.0))
            gallery_changed = False
            if best_id is None or best_score < 0:
                best_id = self.next_id
                self.next_id += 1
                prototype = self._normalise(detection["embedding"])
                self.identities[best_id] = {
                    "class": detection["class"],
                    "prototype": prototype,
                    "gallery": [prototype.copy()],
                    "confidence": confidence,
                    "bbox": detection["bbox"],
                    "active": True,
                    "missing": 0,
                    "samples": 1,
                    "last_seen": datetime.now().isoformat(),
                }
                gallery_changed = True
                logger.info("Nowa tożsamość ReID: #%s (%s)", best_id, detection["class"])
            else:
                available_ids.remove(best_id)
                identity = self.identities[best_id]
                new_embedding = self._normalise(detection["embedding"])
                prototype_weight = 0.60 if confidence > identity["confidence"] + 0.05 else 0.10
                new_prototype = (1.0 - prototype_weight) * identity["prototype"] + prototype_weight * new_embedding
                identity["prototype"] = self._normalise(new_prototype)
                identity["confidence"] = max(identity["confidence"], confidence)

                gallery = identity.setdefault("gallery", [])
                max_gallery_similarity = max(
                    (self._similarity(new_embedding, gallery_embedding) for gallery_embedding in gallery),
                    default=0.0,
                )
                if max_gallery_similarity < 0.82 and confidence >= 0.50:
                    if len(gallery) >= self.max_gallery_size:
                        gallery.pop(0)
                    gallery.append(new_embedding.copy())
                    gallery_changed = True

                identity["bbox"] = detection["bbox"]
                identity["active"] = True
                identity["missing"] = 0
                identity["samples"] += 1
                identity["last_seen"] = datetime.now().isoformat()

            identity = self.identities[best_id]
            if gallery_changed:
                embedding_path = self._persist_identity(best_id, force=True)
            else:
                embedding_path = self._persist_identity(best_id)

            assigned_ids.add(best_id)
            detection["instance_id"] = best_id
            detection["reid_similarity"] = best_score if best_score >= 0 else 1.0
            if embedding_path:
                detection["embedding_path"] = embedding_path
            assignments.append(detection)

        for identity_id, identity in self.identities.items():
            if identity_id not in assigned_ids:
                identity["missing"] += 1
                identity["active"] = False

        return assignments
