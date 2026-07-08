#!/usr/bin/env python3
"""
Główny punkt wejścia systemu.
Użycie:
    python main.py [--mode MODE] [--image PATH]
Tryby:
    gui        – uruchom interfejs graficzny (domyślnie)
    analyze    – analiza pojedynczego obrazu z ID instancji
    camera     – test z kamery na żywo
    add-all    – dodaj wszystkie obiekty z obrazu jako wzorce
"""
import argparse
import sys
import cv2
import logging
from pathlib import Path

# Dodaj katalog src do ścieżki (umożliwia import modułów)
sys.path.insert(0, str(Path(__file__).parent))

from detector import ObjectDetector
from extractor import FeatureExtractor
from database import PatternsDatabase
from matcher import ObjectMatcher
from visualizer import Visualizer
from camera import CameraManager
from analyzer import ImageAnalyzer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_image_mode(image_path: str):
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

def camera_test_mode():
    """Test rozpoznawania na żywo z kamery."""
    detector = ObjectDetector()
    extractor = FeatureExtractor()
    database = PatternsDatabase()
    matcher = ObjectMatcher(database)
    visualizer = Visualizer()
    cam = CameraManager()

    if not cam.open():
        logger.error("Nie można otworzyć kamery")
        return

    logger.info("Kamera uruchomiona. Naciśnij 'q', aby zakończyć.")
    while True:
        ret, frame = cam.read_frame()
        if not ret:
            break

        # Wykrywanie
        detections = detector.detect(frame)

        # Rozpoznawanie
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            crop = frame[y1:y2, x1:x2]
            emb = extractor.extract(crop, det['class'])
            if emb is not None:
                match, sim = matcher.match(emb, det['class'])
                det['match'] = match or 'UNKNOWN'
                det['similarity'] = sim
            else:
                det['match'] = 'UNKNOWN'
                det['similarity'] = 0.0

        # Wizualizacja
        annotated = visualizer.draw_detections(frame, detections)
        cv2.imshow("Test z kamery", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.close()
    cv2.destroyAllWindows()

def add_all_patterns_mode(image_path: str):
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
    parser.add_argument('--mode', choices=['gui','analyze','camera','add-all'],
                        default='gui', help='Tryb działania')
    parser.add_argument('--image', type=str, help='Ścieżka do obrazu (dla analyze/add-all)')
    args = parser.parse_args()

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
        camera_test_mode()
    elif args.mode == 'add-all':
        if not args.image:
            logger.error("Podaj --image dla trybu add-all")
            return
        add_all_patterns_mode(args.image)

if __name__ == "__main__":
    main()