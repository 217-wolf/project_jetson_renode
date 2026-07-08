"""Moduł analizy pojedynczego obrazu - identyfikacja instancji."""
#Biblioteki zewnetrzne
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import cv2
#Biblioteki wewnetrzne pythona
import logging, yaml
from collections import defaultdict
from typing import List, Dict

logger = logging.getLogger(__name__) #do debugu

class ImageAnalyzer:
    """Analizuje pojedynczy obraz - wykrywa i identyfikuje instancje."""
    
    def __init__(self, detector, extractor, config_path: str = "config.yaml"):
        self.detector = detector
        self.extractor = extractor
        with open(config_path) as file:
            config = yaml.safe_load(file)
        self.cluster_config = config['clustering']
    
    def analyze(self, image_path: str) -> tuple:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Nie można wczytać: {image_path}")
        return self.analyze_image(image) #Zwraca: (obraz_z_adnotacjami, lista_detekcji)
    
    
    def analyze_image(self, image: np.ndarray) -> tuple:
        """
        Analizuje obraz w pamięci.
        
        Args:
            image: Obraz w formacie numpy array (BGR)
            
        Returns:
            (obraz_z_adnotacjami, lista_detekcji)
        """
        # Wykryj obiekty
        detections = self.detector.detect(image)
        
        # Wyekstrahuj cechy - postac osadzenia
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            crop = image[y1:y2, x1:x2]
            detection['embedding'] = self.extractor.extract(crop, detection['class'])
        
        # Usuń detekcje bez embeddingów
        detections = [d for d in detections if d.get('embedding') is not None]
        
        # Przydziel ID instancji
        self._assign_instance_ids(detections)
        
        return image, detections
    
    def _assign_instance_ids(self, detections: List[Dict]):
        """Przydziel unikalne ID na podstawie podobieństwa wizualnego."""
        # Grupuj według klasy
        by_class = defaultdict(list)
        for detection in detections:
            by_class[detection['class']].append(detection)
        
        # Klasteryzuj każdą klasę
        for class_name, items in by_class.items():
            n = len(items)
            if n == 1:
                items[0]['instance_id'] = 1
            elif n > 1:
                embeddings = np.array([item['embedding'] for item in items])
                
                clustering = AgglomerativeClustering(
                    n_clusters = None,
                    distance_threshold = self.cluster_config['distance_threshold'],
                    metric = self.cluster_config['metric'],
                    linkage = self.cluster_config['linkage']
                )
                
                labels = clustering.fit_predict(embeddings)
                
                # Mapuj etykiety na czytelne ID
                unique_labels = sorted(set(labels))
                label_to_id = {lab: i + 1 for i, lab in enumerate(unique_labels)}
                
                for detection, lab in zip(items, labels):
                    detection['instance_id'] = label_to_id[lab]
        
        logger.debug(f"Przydzielono ID dla {len(detections)} obiektów")