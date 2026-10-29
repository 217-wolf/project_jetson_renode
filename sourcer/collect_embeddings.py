#!/usr/bin/env python3
"""Zbieraj embeddingi z kamery i zapisuj uporządkowany stan ReID."""

import logging
import threading
import time
from pathlib import Path

import cv2
import yaml

from camera import CameraManager
from detector import ObjectDetector
from extractor import FeatureExtractor
from reid_tracker import EmbeddingStore, PersistentReIDTracker
from visualizer import Visualizer

logger = logging.getLogger("collect_embeddings")


class EmbeddingCollector:
    """Cykl kamery z ReID i kontrolowanym zapisem do ``patterns_database``."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, encoding="utf-8") as file:
            config = yaml.safe_load(file)

        self.cam = CameraManager(config_path)
        self.detector = ObjectDetector(config_path)
        self.extractor = FeatureExtractor(config_path)
        self.visualizer = Visualizer()
        self.embedding_store = EmbeddingStore(Path(config["patterns"]["database_path"]))
        self.tracker = PersistentReIDTracker(store=self.embedding_store)
        self._stop_event = threading.Event()
        self._thread = None

    def run(self) -> None:
        if not self.cam.open():
            logger.error("Nie można otworzyć kamery")
            return

        logger.info("Rozpoczynam zbieranie embeddingów. Q lub Esc kończy pracę.")
        try:
            while not self._stop_event.is_set():
                ok, frame = self.cam.read_frame()
                if not ok:
                    time.sleep(0.01)
                    continue

                trackable = []
                for detection in self.detector.detect(frame):
                    x1, y1, x2, y2 = detection["bbox"]
                    height, width = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    embedding = self.extractor.extract(frame[y1:y2, x1:x2], detection["class"])
                    if embedding is None:
                        continue
                    detection["bbox"] = (x1, y1, x2, y2)
                    detection["embedding"] = embedding
                    trackable.append(detection)

                tracked = self.tracker.update(trackable)
                annotated = self.visualizer.draw_instance_ids(frame, tracked)
                cv2.imshow("Collect Embeddings", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
        finally:
            self.tracker.flush()
            self.cam.close()
            cv2.destroyAllWindows()
            self._log_summary()

    def _log_summary(self) -> None:
        summary = self.embedding_store.get_summary()
        if not summary:
            logger.info("Brak zapisanych tożsamości ReID")
            return
        for class_name, counts in sorted(summary.items()):
            logger.info(
                "%s: %s tożsamości, %s widoków galerii",
                class_name,
                counts["identities"],
                counts["gallery_samples"],
            )

    def start(self, threaded: bool = True) -> None:
        self._stop_event.clear()
        if not threaded:
            self.run()
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)


if __name__ == "__main__":
    EmbeddingCollector().run()
