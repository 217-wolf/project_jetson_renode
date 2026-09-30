#!/usr/bin/env python3
"""Uruchomienie rozpoznawania: istniejące wzorce w bazie albo śledzenie ReID."""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import cv2

from camera import CameraManager
from database import PatternsDatabase
from detector import ObjectDetector
from extractor import FeatureExtractor
from matcher import ObjectMatcher
from reid_tracker import EmbeddingStore, PersistentReIDTracker
from visualizer import Visualizer

logger = logging.getLogger(__name__)


class IdentificationMode(str, Enum):
    DATABASE = "database"
    REID = "reid"
    AUTO = "auto"


def _resolve_config_path(config_path: str) -> str:
    path = Path(config_path)
    if path.is_file():
        return str(path.resolve())
    here = Path(__file__).resolve().parent
    for candidate in (Path.cwd() / config_path, here / config_path, here.parent / config_path):
        if candidate.is_file():
            return str(candidate.resolve())
    return config_path


class RecognitionLauncher:
    """Jedna pętla kamery/wideo z wybranym sposobem identyfikacji.

    ``database`` — dopasowanie do wzorców w ``patterns_database/known``.
    ``reid`` — tracker tożsamości, z wczytaniem stanu z ``patterns_database/reid``.
    ``auto`` — baza, gdy są wzorce; w przeciwnym razie ReID.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        identify: Union[str, IdentificationMode] = IdentificationMode.AUTO,
        source: Optional[Union[str, int]] = None,
        reset_reid: bool = False,
        record: bool = True,
        detector: Optional[ObjectDetector] = None,
        extractor: Optional[FeatureExtractor] = None,
        database: Optional[PatternsDatabase] = None,
    ):
        self.config_path = _resolve_config_path(config_path)
        self.requested_mode = IdentificationMode(identify)
        self.source = source
        self.reset_reid = reset_reid
        self.record = record

        self.detector = detector or ObjectDetector(self.config_path)
        self.extractor = extractor or FeatureExtractor(self.config_path)
        self.database = database or PatternsDatabase(self.config_path)
        self.matcher = ObjectMatcher(self.database, self.config_path)
        self.visualizer = Visualizer()
        self.store = EmbeddingStore(self.database.reid_path)
        self.tracker = PersistentReIDTracker(store=self.store)
        self.active_mode = self.requested_mode

    def has_known_patterns(self) -> bool:
        return self.database.get_statistics()["total_patterns"] > 0

    def has_reid_identities(self) -> bool:
        return any(counts["identities"] > 0 for counts in self.store.get_summary().values())

    def resolve_mode(self) -> IdentificationMode:
        if self.requested_mode == IdentificationMode.DATABASE:
            if not self.has_known_patterns():
                logger.warning("Brak wzorców w bazie — przełączam identyfikację na ReID")
                return IdentificationMode.REID
            return IdentificationMode.DATABASE

        if self.requested_mode == IdentificationMode.REID:
            return IdentificationMode.REID

        if self.has_known_patterns():
            logger.info("Auto: znaleziono wzorce w bazie — używam dopasowania do osadzeń")
            return IdentificationMode.DATABASE

        if self.has_reid_identities():
            logger.info("Auto: brak wzorców, wczytuję zapisane tożsamości ReID")
        else:
            logger.info("Auto: brak wzorców i zapisu ReID — startuję nową sesję ReID")
        return IdentificationMode.REID

    def _prepare_reid(self) -> None:
        if self.reset_reid:
            self.tracker.reset()
            self.store.clear()
            logger.info("Zresetowano tożsamości ReID na dysku")
            return
        loaded = self.tracker.load_from_store()
        if loaded:
            logger.info("Przywrócono %s tożsamości ReID, kolejny ID: %s", loaded, self.tracker.next_id)

    def _annotate_frame(self, frame, detections):
        if self.active_mode == IdentificationMode.DATABASE:
            annotated = self.visualizer.draw_detections(frame, detections)
        else:
            annotated = self.visualizer.draw_instance_ids(frame, detections)
        label = "tryb: baza wzorcow" if self.active_mode == IdentificationMode.DATABASE else "tryb: ReID"
        cv2.putText(annotated, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(annotated, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return annotated

    def _identify(self, frame, detections: List[dict]) -> List[dict]:
        trackable = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            height, width = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            detection["bbox"] = (x1, y1, x2, y2)
            if x2 <= x1 or y2 <= y1:
                detection["match"] = "UNKNOWN"
                detection["similarity"] = 0.0
                continue

            embedding = self.extractor.extract(frame[y1:y2, x1:x2], detection["class"])
            if embedding is None:
                detection["match"] = "UNKNOWN"
                detection["similarity"] = 0.0
                continue

            detection["embedding"] = embedding
            if self.active_mode == IdentificationMode.DATABASE:
                match, similarity = self.matcher.match(embedding, detection["class"])
                detection["match"] = match or "UNKNOWN"
                detection["similarity"] = similarity
            else:
                trackable.append(detection)

        if self.active_mode == IdentificationMode.REID:
            self.tracker.update(trackable)
        return detections

    def _open_source(self):
        if self.source is not None:
            capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                logger.error("Nie można otworzyć źródła wideo: %s", self.source)
                return None
            logger.info("Otworzono plik wideo: %s", self.source)
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            return capture.read, capture.release, source_fps, f"Rozpoznawanie: {self.source}"

        camera = CameraManager(self.config_path)
        if not camera.open():
            logger.error("Nie można otworzyć kamery")
            return None
        source_fps = camera.cap.get(cv2.CAP_PROP_FPS) if camera.cap is not None else 0
        logger.info("Kamera uruchomiona")
        return camera.read_frame, camera.close, source_fps, "Rozpoznawanie z kamery"

    def run(self) -> bool:
        self.active_mode = self.resolve_mode()
        if self.active_mode == IdentificationMode.REID:
            self._prepare_reid()

        opened = self._open_source()
        if opened is None:
            return False
        read_func, close_func, source_fps, window_title = opened
        fps = source_fps if source_fps and source_fps > 1 else 20.0

        writer = None
        output_path = None
        if self.record:
            recordings_dir = self.database.db_path.parent / "recordings"
            recordings_dir.mkdir(parents=True, exist_ok=True)
            output_path = recordings_dir / f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        logger.info("Identyfikacja: %s. Q lub Esc kończy pracę.", self.active_mode.value)
        try:
            while True:
                ok, frame = read_func()
                if not ok or frame is None:
                    logger.info("Koniec strumienia wideo / brak klatki")
                    break

                detections = self._identify(frame, self.detector.detect(frame))
                annotated = self._annotate_frame(frame, detections)

                if self.record:
                    if writer is None:
                        height, width = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                        if not writer.isOpened():
                            logger.error("Nie można utworzyć pliku nagrania: %s", output_path)
                            writer = None
                        else:
                            logger.info("Nagrywanie do: %s", output_path)
                    if writer is not None:
                        writer.write(annotated)

                cv2.imshow(window_title, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
        finally:
            if self.active_mode == IdentificationMode.REID:
                self.tracker.flush()
            if writer is not None:
                writer.release()
            close_func()
            cv2.destroyAllWindows()
        return True


def launch_recognition(
    identify: Union[str, IdentificationMode] = IdentificationMode.AUTO,
    source: Optional[Union[str, int]] = None,
    reset_reid: bool = False,
    record: bool = True,
    config_path: str = "config.yaml",
    detector: Optional[ObjectDetector] = None,
    extractor: Optional[FeatureExtractor] = None,
    database: Optional[PatternsDatabase] = None,
) -> bool:
    """Publiczny punkt startu systemu rozpoznawania. Zwraca False, gdy źródło się nie otworzy."""
    return RecognitionLauncher(
        config_path=config_path,
        identify=identify,
        source=source,
        reset_reid=reset_reid,
        record=record,
        detector=detector,
        extractor=extractor,
        database=database,
    ).run()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Uruchom rozpoznawanie: baza wzorców albo ReID")
    parser.add_argument("--identify", choices=["auto", "database", "reid"], default="auto")
    parser.add_argument("--video", type=str, help="Plik wideo; bez argumentu używana jest kamera")
    parser.add_argument("--reset", action="store_true", help="Wyczyść zapisane tożsamości ReID")
    parser.add_argument("--no-record", action="store_true", help="Nie nagrywaj wyniku")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    ok = launch_recognition(
        identify=args.identify,
        source=args.video,
        reset_reid=args.reset,
        record=not args.no_record,
        config_path=args.config,
    )
    raise SystemExit(0 if ok else 1)
