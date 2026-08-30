#!/usr/bin/env python3
"""
Główny punkt wejścia systemu.
Użycie:
    python main.py [--mode MODE] [--image PATH] [--video PATH]
Tryby:
    gui        - uruchom interfejs graficzny (domyślnie)
    analyze    - analiza pojedynczego obrazu z ID instancji
    camera     - test na żywo z kamery lub z pliku wideo (jeśli podano --video)
    add-all    - dodaje wszystkie obiekty z obrazu jako wzorce
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional, Union

import cv2

# Dodaj katalog sourcer do ścieżki (umożliwia import modułów)
sys.path.insert(0, str(Path(__file__).parent))

# Import modułów
from analyzer import ImageAnalyzer
from camera import CameraManager
from database import PatternsDatabase
from detector import ObjectDetector
from extractor import FeatureExtractor
from matcher import ObjectMatcher
from reid_tracker import PersistentReIDTracker
from visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def analyze_image_mode(image_path: str) -> None:
    """Tryb analizy pojedynczego obrazu (ID instancji)."""
    detector = ObjectDetector()
    extractor = FeatureExtractor()
    analyzer = ImageAnalyzer(detector, extractor)
    visualizer = Visualizer()

    logger.info(f"Analiza obrazu: {image_path}")
    img, detections = analyzer.analyze(image_path)
    annotated = visualizer.draw_instance_ids(img, detections)

    out_path = f"analiza_{Path(image_path).stem}.jpg"
    cv2.imwrite(out_path, annotated)
    logger.info(f"Zapisano wynik: {out_path}")

    # Podsumowanie
    from collections import defaultdict
    summary = defaultdict(list)
    for d in detections:
        summary[d['class']].append(d['instance_id'])
    for cls, ids in sorted(summary.items()):
        unique_ids = sorted(set(ids))
        logger.info(f"{cls}: {len(ids)} obiektów (ID: {', '.join(map(str, unique_ids))})")

    cv2.imshow("Analiza obrazu", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def camera_test_mode(source: Optional[Union[str, int]] = None, reset_tracker: bool = False) -> None:
    """
    Test rozpoznawania na żywo z kamery lub z pliku wideo.
    :param source: ścieżka do pliku wideo lub indeks kamery – jeśli None, używa CameraManager.
    :param reset_tracker: jeśli True, czyści wcześniejszą historię ID i zaczyna numerację od 1.
    """
    detector = ObjectDetector()
    extractor = FeatureExtractor()
    database = PatternsDatabase()
    matcher = ObjectMatcher(database)
    reid_tracker = PersistentReIDTracker()
    if reset_tracker:
        reid_tracker.reset()
    visualizer = Visualizer()

    # Wybór źródła (plik wideo lub kamera)
    if source is not None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Nie można otworzyć źródła wideo: {source}")
            return
        read_func = cap.read
        close_func = cap.release
        logger.info(f"Otworzono plik wideo: {source}")
    else:
        cam = CameraManager()
        if not cam.open():
            logger.error("Nie można otworzyć kamery")
            return
        read_func = cam.read_frame
        close_func = cam.close
        logger.info("Kamera uruchomiona.")

    logger.info("Naciśnij 'Q' lub 'Esc', aby zakończyć.")
    while True:
        ret, frame = read_func()
        if not ret:
            logger.info("Koniec strumienia wideo / brak klatki.")
            break

        # 1. Wykrywanie obiektów
        detections = detector.detect(frame)

        # 2. Ekstrakcja cech i dopasowanie do wzorców
        trackable = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            crop = frame[y1:y2, x1:x2]
            emb = extractor.extract(crop, det['class'])
            if emb is not None:
                det['embedding'] = emb
                trackable.append(det)
                match, sim = matcher.match(emb, det['class'])
                det['match'] = match or 'UNKNOWN'
                det['similarity'] = sim
            else:
                det['match'] = 'UNKNOWN'
                det['similarity'] = 0.0

        # 3. Aktualizacja trackera ReID (dokładnie raz na klatkę poza pętlą for)
        reid_tracker.update(trackable)

        # 4. Wizualizacja wyników
        annotated = visualizer.draw_detections(frame, detections)
        cv2.imshow("Test z kamery / wideo", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # 'Q' lub 'Esc'
            break

    close_func()
    cv2.destroyAllWindows()


def add_all_patterns_mode(image_path: str) -> None:
    """Dodaj wszystkie wykryte obiekty z obrazu jako wzorce."""
    detector = ObjectDetector()
    extractor = FeatureExtractor()
    database = PatternsDatabase()

    img = cv2.imread(image_path)

    if img is None:
        logger.error(f"Nie można wczytać: {image_path}")
        return

    detections = detector.detect(img)
    added = 0
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        crop = img[y1:y2, x1:x2]
        emb = extractor.extract(crop, det['class'])

        if emb is not None:
            name = f"{det['class']}_{i+1:03d}"
            database.add_pattern(name, det['class'], emb, det['confidence'])
            added += 1

    logger.info(f"Dodano {added} wzorców z obrazu {image_path}")


def main():
    parser = argparse.ArgumentParser(description="System rozpoznawania obiektów")
    parser.add_argument(
        '--mode',
        choices=['gui', 'analyze', 'camera', 'add-all'],
        default='gui',
        help='Tryb działania'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Ścieżka do obrazu (dla analyze/add-all)'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Ścieżka do pliku wideo (dla trybu camera) – jeśli pominięto, używana jest kamera'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Wyczyść dotychczas zebrane tożsamości i zacznij numerację ID od nowa (od #1)'
    )
    args = parser.parse_args()

    # Jeśli podano plik wideo, a tryb to domyślny 'gui', automatycznie przełącz na odtwarzanie wideo
    if args.video and args.mode == 'gui':
        args.mode = 'camera'

    if args.mode == 'gui':

        from gui import MainApplication
        
        app = MainApplication()
        app.run()

    elif args.mode == 'analyze':

        if not args.image:
            logger.error("Podaj --image dla trybu analyze")

            return

        analyze_image_mode(args.image)

    elif args.mode == 'camera':
        camera_test_mode(args.video, reset_tracker=args.reset)

    elif args.mode == 'add-all':
        if not args.image:
            logger.error("Podaj --image dla trybu add-all")
            return
        add_all_patterns_mode(args.image)


if __name__ == "__main__":
    main()
