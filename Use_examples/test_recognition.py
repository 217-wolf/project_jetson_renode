#!/usr/bin/env python3
"""Przykład: Test rozpoznawania."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sourcer.detector import ObjectDetector
from sourcer.extractor import FeatureExtractor
from sourcer.database import PatternsDatabase
from sourcer.matcher import ObjectMatcher
from sourcer.visualizer import Visualizer
import cv2

def main():
    # Inicjalizacja
    detector = ObjectDetector()
    extractor = FeatureExtractor()
    database = PatternsDatabase()
    matcher = ObjectMatcher(database)
    visualizer = Visualizer()
    
    # Wczytaj obraz testowy
    image = cv2.imread("examples/test_image.jpg")
    
    # Wykryj i rozpoznaj
    detections = detector.detect(image)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        crop = image[y1:y2, x1:x2]
        
        # Ekstrahuj i dopasuj
        embedding = extractor.extract(crop, det['class'])
        if embedding is not None:
            match, similarity = matcher.match(embedding, det['class'])
            det['match'] = match or 'UNKNOWN'
            det['similarity'] = similarity
    
    # Wizualizuj
    result = visualizer.draw_detections(image, detections)
    
    # Zapisz i pokaż
    cv2.imwrite("result.jpg", result)
    cv2.imshow("Wynik", result)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()