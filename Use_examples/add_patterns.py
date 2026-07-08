#!/usr/bin/env python3
"""Przykład: Dodawanie wzorców do bazy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sourcer.detector import ObjectDetector
from sourcer.extractor import FeatureExtractor
from sourcer.database import PatternsDatabase
import cv2

def main():
    # Inicjalizacja komponentów
    detector = ObjectDetector()
    extractor = FeatureExtractor()
    database = PatternsDatabase()
    
    # Wczytaj obraz
    image_path = "examples/test_image.jpg"
    image = cv2.imread(image_path)
    
    # Wykryj obiekty
    detections = detector.detect(image)
    
    # Dodaj każdy obiekt jako wzorzec
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        crop = image[y1:y2, x1:x2]
        
        # Ekstrahuj cechy
        embedding = extractor.extract(crop, det['class'])
        if embedding is not None:
            # Nadaj nazwę
            name = f"{det['class']}_{i+1:03d}"
            
            # Dodaj do bazy
            database.add_pattern(name, det['class'], embedding, det['confidence'])
            print(f"Dodano wzorzec: {name}")
    
    # Pokaż statystyki
    stats = database.get_statistics()
    print(f"\nStatystyki bazy:")
    print(f"Wzorców: {stats['total_patterns']}")
    print(f"Klas: {stats['total_classes']}")

if __name__ == "__main__":
    main()