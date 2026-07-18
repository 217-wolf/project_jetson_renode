#!/usr/bin/env python3
"""Prosty logger CSV do zapisu zdarzeń embeddingów.

Zawiera klasę `EmbeddingLogger` z metodami do zapisu wpisów oraz czytania podsumowań.
"""
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class EmbeddingLogger:
    def __init__(self, csv_path: str = 'collected_embeddings/embeddings_log.csv'):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Jeśli plik nie istnieje - stwórz nagłówek
        if not self.path.exists():
            with self.path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'class', 'instance_id', 'filepath', 'confidence'])

    def log_entry(self, class_name: str, instance_id: int, filepath: str, confidence: float = None, timestamp: str = None):
        ts = timestamp or datetime.now().isoformat()
        with self.path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([ts, class_name, instance_id, filepath, '' if confidence is None else float(confidence)])

    def get_recent(self, n: int = 100) -> List[Dict]:
        rows = []
        if not self.path.exists():
            return rows
        with self.path.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows[-n:]

    def get_summary(self) -> Dict[str, int]:
        counts = {}
        if not self.path.exists():
            return counts
        with self.path.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls = row.get('class', 'unknown')
                counts[cls] = counts.get(cls, 0) + 1
        return counts
