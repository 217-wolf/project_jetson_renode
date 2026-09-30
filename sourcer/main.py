#!/usr/bin/env python3
"""
Główny punkt wejścia systemu.
Użycie:
    python main.py [--mode MODE] [--image PATH] [--video PATH] [--identify MODE]
Tryby:
    gui        - uruchom interfejs graficzny (domyślnie)
    analyze    - analiza pojedynczego obrazu z ID instancji
    camera     - test na żywo z kamery lub z pliku wideo (jeśli podano --video)
    add-all    - dodaje wszystkie obiekty z obrazu jako wzorce
Identyfikacja (--identify):
    auto       - wzorce z bazy, jeśli istnieją; w przeciwnym razie ReID
    database   - dopasowanie do osadzeń w patterns_database/known
    reid       - śledzenie tożsamości z patterns_database/reid
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
from database import PatternsDatabase
from detector import ObjectDetector
from extractor import FeatureExtractor
from launcher import launch_recognition
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


def camera_test_mode(
    source: Optional[Union[str, int]] = None,
    reset_tracker: bool = False,
    identify: str = "auto",
) -> None:
    """Rozpoznawanie na żywo z kamery lub pliku: baza wzorców albo ReID."""
    launch_recognition(
        identify=identify,
        source=source,
        reset_reid=reset_tracker,
        record=True,
    )


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
    parser.add_argument(
        '--identify',
        choices=['auto', 'database', 'reid'],
        default='auto',
        help='Sposób identyfikacji: baza wzorców, ReID albo auto (baza jeśli są wzorce)'
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
        camera_test_mode(args.video, reset_tracker=args.reset, identify=args.identify)

    elif args.mode == 'add-all':
        if not args.image:
            logger.error("Podaj --image dla trybu add-all")
            return
        add_all_patterns_mode(args.image)


if __name__ == "__main__":
    main()
