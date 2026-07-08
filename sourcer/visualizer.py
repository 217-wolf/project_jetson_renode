"""Moduł wizualizacji wyników detekcji."""
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """Wizualizacja wyników detekcji na obrazie."""
    
    def __init__(self):
        self.colors = {}
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det.get('confidence', 0.0)
            match_name = det.get('match', 'UNKNOWN')
            similarity = det.get('similarity', 0.0)
            
            # Wybierz kolor
            if match_name and match_name != 'UNKNOWN':
                color = self._get_color(match_name)
                status = "MATCH"
            else:
                color = (0, 0, 255)  # Czerwony dla nieznanych
                status = "UNKNOWN"
            
            # Rysuj ramkę
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Przygotuj tekst
            if status == "MATCH":
                text = f"{match_name} ({class_name}) | {confidence:.2f} | sim:{similarity:.3f}"
            else:
                text = f"UNKNOWN ({class_name}) | {confidence:.2f}"
            
            # Rysuj tło tekstu
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated
    
    def draw_instance_ids(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Narysuj detekcje z unikalnymi ID instancji.
        
        Args:
            image: Obraz wejściowy
            detections: Lista detekcji z polami: bbox, class, instance_id, confidence
            
        Returns:
            Obraz z narysowanymi ramkami i ID
        """
        annotated = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            instance_id = detection.get('instance_id', 0)
            confidence = detection.get('confidence', 0.0)
            
            color = self._color_for_id(instance_id)
            label = f"{class_name}_{instance_id} | {confidence:.2f}"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return annotated
    
    def _get_color(self, name: str) -> Tuple[int, int, int]:
        """Generuj stabilny kolor na podstawie nazwy."""
        if name not in self.colors:
            # Hash nazwy do wartości hue
            hue = hash(name) % 180
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            self.colors[name] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        return self.colors[name]
    
    @staticmethod
    def _color_for_id(instance_id: int) -> Tuple[int, int, int]:
        """Generuj kolor na podstawie ID instancji."""
        hue = (instance_id * 43) % 180
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))