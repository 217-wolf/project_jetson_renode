"""Uporządkowana baza znanych wzorców obiektów."""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "unnamed"


class PatternsDatabase:
    """Baza wzorców w układzie ``known/<klasa>/<nazwa>/``.

    Katalog ``reid`` pozostaje własnością :class:`EmbeddingStore` i nigdy nie
    jest przez tę klasę kasowany ani nadpisywany.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, encoding="utf-8") as file:
            config = yaml.safe_load(file)

        self.db_path = Path(config["patterns"]["database_path"])
        self.known_root = self.db_path
        self.known_root.mkdir(parents=True, exist_ok=True)
        self.patterns: Dict[str, List[Dict]] = {}
        self._load()

    def _pattern_dir(self, class_name: str, name: str) -> Path:
        return self.known_root / _safe_component(class_name) / _safe_component(name)

    def _load(self) -> None:
        """Wczytaj wszystkie kompletne wzorce z katalogu."""
        for class_dir in self.known_root.iterdir():
            if not class_dir.is_dir():
                continue
            for pattern_dir in class_dir.iterdir():
                metadata_file = pattern_dir / "metadata.json"
                prototype_file = pattern_dir / "prototype.npy"
                if not pattern_dir.is_dir() or not metadata_file.exists() or not prototype_file.exists():
                    logger.warning("Pomijam niekompletny wzorzec: %s", pattern_dir)
                    continue
                try:
                    with metadata_file.open(encoding="utf-8") as file:
                        metadata = json.load(file)
                    class_name = metadata["class"]
                    self.patterns.setdefault(class_name, []).append({
                        "name": metadata["name"],
                        "embedding": np.load(prototype_file),
                        "confidence": float(metadata.get("confidence", 0.0)),
                        "timestamp": metadata.get("timestamp", ""),
                        "metadata": metadata.get("metadata", {}),
                    })
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                    logger.warning("Nie można wczytać wzorca %s: %s", pattern_dir, error)

        logger.info("Wczytano %s wzorców", sum(len(items) for items in self.patterns.values()))

    def _write_pattern(self, pattern: Dict, class_name: str) -> None:
        pattern_dir = self._pattern_dir(class_name, pattern["name"])
        pattern_dir.mkdir(parents=True, exist_ok=True)
        np.save(pattern_dir / "prototype.npy", np.asarray(pattern["embedding"], dtype=np.float32))
        payload = {
            "schema_version": 1,
            "name": pattern["name"],
            "class": class_name,
            "confidence": float(pattern["confidence"]),
            "timestamp": pattern["timestamp"],
            "metadata": pattern.get("metadata", {}),
        }
        with (pattern_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def add_pattern(
        self,
        name: str,
        class_name: str,
        embedding: np.ndarray,
        confidence: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Dodaj lub zastąp wzorzec o podanej nazwie i klasie."""
        class_patterns = self.patterns.setdefault(class_name, [])
        timestamp = datetime.now().isoformat()
        for pattern in class_patterns:
            if pattern["name"] == name:
                pattern.update({
                    "embedding": np.asarray(embedding, dtype=np.float32),
                    "confidence": confidence,
                    "timestamp": timestamp,
                    "metadata": metadata or {},
                })
                self._write_pattern(pattern, class_name)
                logger.info("Zaktualizowano wzorzec: %s (%s)", name, class_name)
                return True

        pattern = {
            "name": name,
            "embedding": np.asarray(embedding, dtype=np.float32),
            "confidence": confidence,
            "timestamp": timestamp,
            "metadata": metadata or {},
        }
        class_patterns.append(pattern)
        self._write_pattern(pattern, class_name)
        logger.info("Dodano wzorzec: %s (%s)", name, class_name)
        return True

    def remove_pattern(self, name: str, class_name: Optional[str] = None) -> bool:
        classes = [class_name] if class_name else list(self.patterns)
        removed = False
        for current_class in classes:
            patterns = self.patterns.get(current_class, [])
            remaining = [pattern for pattern in patterns if pattern["name"] != name]
            if len(remaining) == len(patterns):
                continue
            shutil.rmtree(self._pattern_dir(current_class, name), ignore_errors=True)
            if remaining:
                self.patterns[current_class] = remaining
            else:
                self.patterns.pop(current_class, None)
                class_dir = self.known_root / _safe_component(current_class)
                if class_dir.exists() and not any(class_dir.iterdir()):
                    class_dir.rmdir()
            removed = True

        if removed:
            logger.info("Usunięto wzorzec: %s", name)
        return removed

    def get_patterns_klasa(self, class_name: str) -> List[Dict]:
        return self.patterns.get(class_name, [])

    def get_all_patterns(self) -> List[Dict]:
        return sorted(
            [
                {
                    "name": pattern["name"],
                    "class": class_name,
                    "confidence": pattern["confidence"],
                    "timestamp": pattern["timestamp"],
                    "metadata": pattern.get("metadata", {}),
                }
                for class_name, patterns in self.patterns.items()
                for pattern in patterns
            ],
            key=lambda item: (item["class"], item["name"]),
        )

    def get_statistics(self) -> Dict:
        return {
            "total_patterns": sum(len(items) for items in self.patterns.values()),
            "total_classes": len(self.patterns),
            "classes": {class_name: len(items) for class_name, items in self.patterns.items()},
        }

    def clear(self) -> None:
        """Usuń wyłącznie znane wzorce; zachowaj zapis ReID."""
        shutil.rmtree(self.known_root, ignore_errors=True)
        self.known_root.mkdir(parents=True, exist_ok=True)
        self.patterns.clear()
        logger.info("Baza znanych wzorców wyczyszczona")
