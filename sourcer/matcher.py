"""Moduł dopasowywania obiektów do wzorców."""
import numpy as np
from scipy.spatial.distance import cosine
from typing import Optional, Tuple, Dict, List
import yaml
import logging

logger = logging.getLogger(__name__)

class ObjectMatcher:
    """Dopasowuje wykryte obiekty do bazy wzorców."""
    
    def __init__(self, database, config_path: str = "config.yaml"):
        self.database = database
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.thresholds = config['patterns']['matching_threshold']
    
    def match(self, embedding: np.ndarray, class_name: str) -> Tuple[Optional[str], float]:
        """
        Dopasuj embedding do wzorca.
        
        Args:
            embedding: Wektor cech obiektu
            class_name: Klasa obiektu
            
        Returns:
            (nazwa_dopasowania, podobieństwo) - (None, 0.0) jeśli nie dopasowano
        """
        patterns = self.database.get_patterns_klasa(class_name)
        if not patterns:
            logger.debug(f"Brak wzorców dla klasy: {class_name}")
            return None, 0.0
        
        # Wybierz próg dla danej klasy
        threshold = self.thresholds.get(class_name, self.thresholds['default'])
        
        best_match = None
        best_similarity = 0.0
        
        for pattern in patterns:
            similarity = 1 - cosine(embedding.flatten(), pattern['embedding'].flatten())
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = pattern['name']
        
        if best_match:
            logger.debug(f"Dopasowano: {best_match} ({class_name}) - sim: {best_similarity:.4f}")
        else:
            logger.debug(f"Nie dopasowano: {class_name}")
        
        return best_match, best_similarity
    
    def match_batch(self, embeddings: List[np.ndarray], 
                   class_names: List[str]) -> List[Tuple[Optional[str], float]]:
        """
        Dopasuj wiele embeddingów.
        
        Args:
            embeddings: Lista wektorów cech
            class_names: Lista nazw klas
            
        Returns:
            Lista krotek (nazwa_dopasowania, podobieństwo)
        """
        return [self.match(emb, cls) for emb, cls in zip(embeddings, class_names)]