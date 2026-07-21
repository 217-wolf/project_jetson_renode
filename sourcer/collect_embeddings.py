#!/usr/bin/env python3
"""Zbiera embeddingi obiektów z kamery, śledzi instancje i zapisuje wektory cech.

Uruchomienie:
    python collect_embeddings.py

Plik `camera.sh` uruchamia ten skrypt.
"""
import cv2
import time
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine
import threading

from camera import CameraManager
from detector import ObjectDetector
from extractor import FeatureExtractor
from visualizer import Visualizer
from embedding_logger import EmbeddingLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('collect_embeddings')


class SimpleTracker:
    """Lekki tracker oparty na embeddingach + IOU.
    Przechowuje ostatnie osadzenia dla każdej instancji i przydziela stabilne ID.
    """
    def __init__(self, sim_threshold: float = 0.80, iou_threshold: float = 0.4, max_age: int = 30):
        self.next_id = 1
        self.tracks = {}  # id -> track dict
        self.sim_threshold = sim_threshold
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - interArea
        return interArea / union if union > 0 else 0.0

    def update(self, detections):
        """Aktualizuje śledzenie na podstawie nowych detekcji.

        Args:
            detections: lista słowników z kluczami: bbox, class, embedding, confidence
        Returns:
            lista detekcji uzupełnionych o `instance_id`
        """
        assigned = []
        # Przygotuj candidate tracks
        unmatched_tracks = set(self.tracks.keys())

        for det in detections:
            best_id = None
            best_score = -1.0
            for tid, tr in self.tracks.items():
                if tr['class'] != det['class']:
                    continue
                # similarity
                try:
                    sim = 1 - cosine(det['embedding'].flatten(), tr['last_embedding'].flatten())
                except Exception:
                    sim = 0.0
                iou = self._iou(det['bbox'], tr['bbox'])
                # prefer embedding similarity but allow IOU fallback
                score = sim if sim >= self.sim_threshold else (iou if iou >= self.iou_threshold else -1.0)
                if score > best_score:
                    best_score = score
                    best_id = tid

            if best_id is not None and best_score >= 0:
                # aktualizuj istniejący track
                tr = self.tracks[best_id]
                tr['bbox'] = det['bbox']
                tr['last_embedding'] = det['embedding']
                tr['last_seen'] = 0
                tr['hits'] += 1
                tr['embeddings'].append(det['embedding'])
                det['instance_id'] = best_id
                unmatched_tracks.discard(best_id)
            else:
                # nowy track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'class': det['class'],
                    'bbox': det['bbox'],
                    'last_embedding': det['embedding'],
                    'last_seen': 0,
                    'hits': 1,
                    'embeddings': [det['embedding']]
                }
                det['instance_id'] = tid

            assigned.append(det)

        # Aging
        to_delete = []
        for tid in list(self.tracks.keys()):
            if tid in unmatched_tracks:
                self.tracks[tid]['last_seen'] += 1
            if self.tracks[tid]['last_seen'] > self.max_age:
                to_delete.append(tid)

        for tid in to_delete:
            logger.debug(f"Usuwam track {tid} - brak aktywności")
            del self.tracks[tid]

        return assigned


def save_embedding(out_dir: Path, class_name: str, instance_id: int, embedding: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    fname = out_dir / f"{class_name}_{instance_id}_{ts}.npy"
    np.save(str(fname), embedding)
    return fname


class EmbeddingCollector:
    """Klasa zarządzająca cyklem zbierania embeddingów z kamery.
    Umożliwia uruchamianie w wątku i kontrolę stop/start.
    """
    def __init__(self):
        self.cam = CameraManager()
        self.detector = ObjectDetector()
        self.extractor = FeatureExtractor()
        self.visualizer = Visualizer()
        self.tracker = SimpleTracker()
        self.out_root = Path('collected_embeddings')
        self.out_root.mkdir(exist_ok=True)
        self.csv_logger = EmbeddingLogger()
        self._stop_event = threading.Event()
        self._thread = None

    def run(self):
        if not self.cam.open():
            logger.error("Nie można otworzyć kamery")
            return

        logger.info("Rozpoczynam zbieranie embeddingów. Użyj metody stop() by zakończyć.")

        while not self._stop_event.is_set():
            ok, frame = self.cam.read_frame()
            if not ok:
                time.sleep(0.01)
                continue

            detections = self.detector.detect(frame)

            # Wyekstrahuj embeddingi
            cleaned = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                h, w = frame.shape[:2]
                x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2c <= x1c or y2c <= y1c:
                    continue
                crop = frame[y1c:y2c, x1c:x2c]
                emb = self.extractor.extract(crop, det['class'])
                if emb is None:
                    continue
                det['embedding'] = emb
                cleaned.append(det)

            tracked = self.tracker.update(cleaned)

            # Zapisuj embeddingi natychmiast i dodaj pola do wizualizacji
            for det in tracked:
                inst = det['instance_id']
                cls = det['class']
                emb = det['embedding']
                saved = save_embedding(self.out_root / cls, cls, inst, emb)
                det['saved_path'] = str(saved)
                try:
                    self.csv_logger.log_entry(cls, inst, str(saved), det.get('confidence'))
                except Exception:
                    logger.exception('Błąd zapisu do CSV')

            annotated = self.visualizer.draw_instance_ids(frame, tracked)
            cv2.imshow('Collect Embeddings', annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        self.cam.close()
        cv2.destroyAllWindows()

        # Podsumowanie
        counts = {}
        for p in self.out_root.glob('**/*.npy'):
            cls = p.parent.name
            counts[cls] = counts.get(cls, 0) + 1

        if counts:
            logger.info('Zapisane embeddingi:')
            for cls, cnt in counts.items():
                logger.info(f"  {cls}: {cnt}")
        else:
            logger.info('Brak zapisanych embeddingów')

    def start(self, threaded: bool = True):
        self._stop_event.clear()
        if threaded:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self.run, daemon=True)
            self._thread.start()
        else:
            self.run()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)


if __name__ == '__main__':
    collector = EmbeddingCollector()
    collector.run()
