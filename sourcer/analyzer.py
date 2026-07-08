"""Moduł analizy pojedynczego obrazu - identyfikacja instancji."""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from typing import List, Dict
import cv2
import yaml
import logging

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """Analizuje pojedynczy obraz - wykrywa i identyfikuje instancje."""
    
    def __init__(self, detector, extractor, config_path: str = "config.yaml"):
        self.detector = detector
        self.extractor = extractor
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.cluster_config = config['clustering']
    
    def analyze(self, image_path: str) -> tuple:
        """
        Analizuj obraz - nadaj unikalne ID instancjom.
        
        Args:
            image_path: Ścieżka do obrazu
            
        Returns:
            (obraz_z_adnotacjami, lista_detekcji)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Nie można wczytać: {image_path}")
        
        return self.analyze_image(img)
    
    def analyze_image(self, image: np.ndarray) -> tuple:
        """
        Analizuj obraz w pamięci.
        
        Args:
            image: Obraz w formacie numpy array (BGR)
            
        Returns:
            (obraz_z_adnotacjami, lista_detekcji)
        """
        # Wykryj obiekty
        detections = self.detector.detect(image)
        
        # Wyekstrahuj cechy
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            crop = image[y1:y2, x1:x2]
            det['embedding'] = self.extractor.extract(crop, det['class'])
        
        # Usuń detekcje bez embeddingów
        detections = [d for d in detections if d.get('embedding') is not None]
        
        # Przydziel ID instancji
        self._assign_instance_ids(detections)
        
        return image, detections
    
    def _assign_instance_ids(self, detections: List[Dict]):
        """Przydziel unikalne ID na podstawie podobieństwa wizualnego."""
        # Grupuj według klasy
        by_class = defaultdict(list)
        for det in detections:
            by_class[det['class']].append(det)
        
        # Klasteryzuj każdą klasę
        for class_name, items in by_class.items():
            n = len(items)
            if n == 1:
                items[0]['instance_id'] = 1
            elif n > 1:
                embeddings = np.array([it['embedding'] for it in items])
                
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.cluster_config['distance_threshold'],
                    metric=self.cluster_config['metric'],
                    linkage=self.cluster_config['linkage']
                )
                
                labels = clustering.fit_predict(embeddings)
                
                # Mapuj etykiety na czytelne ID
                unique_labels = sorted(set(labels))
                label_to_id = {lab: i + 1 for i, lab in enumerate(unique_labels)}
                
                for det, lab in zip(items, labels):
                    det['instance_id'] = label_to_id[lab]
        
        logger.debug(f"Przydzielono ID dla {len(detections)} obiektów")