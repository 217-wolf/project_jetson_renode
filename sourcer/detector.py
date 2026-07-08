"""Moduł detekcji obiektów YOLO."""
import cv2, torch
from ultralytics import YOLO
import numpy as np

import logging, yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__) #informacja o module - do debugu

class ObjectDetector:
    """Detektor obiektów wykorzystujący model YOLOv8."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as file:
            self.config = yaml.safe_load(file)
        
        self.conf_threshold = self.config['models']['yolo']['confidence_threshold'] #próg przyjęcia obiektu jako rozpoznanego
        self.img_size = self.config['models']['yolo']['image_size']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Wybierz model (TensorRT jeśli dostępny)
        engine_path = Path(self.config['models']['yolo']['engine_path'])
        model_path = self.config['models']['yolo']['path']
        
        if engine_path.exists() and self.device == 'cuda':
            self.model = YOLO(str(engine_path))
            logger.info(f"Załadowano model TensorRT: {engine_path}") # informacja o modelu TensorRT
        else:
            self.model = YOLO(model_path)
            logger.info(f"Załadowano model: {model_path}") # informacja modelu ze ścieżki
        
        if self.device == 'cuda':
            self.model.to('cuda')
        
        self.class_names = self.model.names
        self.name_to_id = {v: k for k, v in self.class_names.items()}
    
    def detect(self, image: np.ndarray, classes: Optional[List[str]] = None) -> List[Dict]:
        """
        Wykrycie obiektów na obrazie.
        
        Args:
            image: Obraz w formacie BGR (numpy array)
            classes: Lista nazw klas do detekcji (None = wszystkie)
            
        Returns:
            Lista słowników z detekcjami: {bbox, class, confidence}
        """
        class_ids = None
        if classes:
            class_ids = [self.name_to_id[klasa] for klasa in classes if klasa in self.name_to_id]
        
        results = self.model(image, classes = class_ids, conf = self.conf_threshold, 
                           imgsz = self.img_size, verbose = False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    cls_name = self.class_names[cls_id]
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'class': cls_name,
                        'confidence': conf,
                        'class_id': cls_id})
        
        logger.debug(f"Wykryto {len(detections)} obiektów") #informacja logu o liczbie wykrytych elementów
        return detections
    
    def get_class_id(self, class_name: str) -> Optional[int]:
        return self.name_to_id.get(class_name)
    
    def get_class_name(self, class_id: int) -> str:
        return self.class_names.get(class_id, "unknown")