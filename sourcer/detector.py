"""Moduł detekcji obiektów YOLO."""
import cv2
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ObjectDetector:
    """Detektor obiektów oparty na YOLOv26."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.conf_threshold = self.config['models']['yolo']['confidence_threshold']
        self.img_size = self.config['models']['yolo']['image_size']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Wybierz model (TensorRT jeśli dostępny)
        engine_path = Path(self.config['models']['yolo']['engine_path'])
        model_path = self.config['models']['yolo']['path']
        
        if engine_path.exists() and self.device == 'cuda':
            self.model = YOLO(str(engine_path))
            logger.info(f"Załadowano model TensorRT: {engine_path}")
        else:
            self.model = YOLO(model_path)
            logger.info(f"Załadowano model: {model_path}")
        
        if self.device == 'cuda':
            self.model.to('cuda')
        
        self.class_names = self.model.names
        self.name_to_id = {v: k for k, v in self.class_names.items()}
    
    def detect(self, image: np.ndarray, classes: Optional[List[str]] = None) -> List[Dict]:
        """
        Wykryj obiekty na obrazie.
        
        Args:
            image: Obraz w formacie BGR (numpy array)
            classes: Lista nazw klas do detekcji (None = wszystkie)
            
        Returns:
            Lista słowników z detekcjami: {bbox, class, confidence}
        """
        class_ids = None
        if classes:
            class_ids = [self.name_to_id[c] for c in classes if c in self.name_to_id]
        
        results = self.model(image, classes=class_ids, 
                           conf=self.conf_threshold, 
                           imgsz=self.img_size, 
                           verbose=False)
        
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
                        'class_id': cls_id
                    })
        
        logger.debug(f"Wykryto {len(detections)} obiektów")
        return detections
    
    def get_class_id(self, class_name: str) -> Optional[int]:
        """Pobierz ID klasy na podstawie nazwy."""
        return self.name_to_id.get(class_name)
    
    def get_class_name(self, class_id: int) -> str:
        """Pobierz nazwę klasy na podstawie ID."""
        return self.class_names.get(class_id, "unknown")