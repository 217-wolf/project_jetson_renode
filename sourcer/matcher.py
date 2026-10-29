"""Moduł dopasowywania obiektów do istniejących wzorców."""

import numpy as np
from scipy.spatial.distance import cosine

from typing import Optional, Tuple, Dict, List
import logging, yaml

logger = logging.getLogger(__name__) #informacja o module


class ObjectMatcher:
    """Dopasowuje wykryte obiekty do bazy wzorców."""

    def __init__(self, database, config_path: str = "config.yaml"):
        self.database = database
        with open(config_path) as file:
            config = yaml.safe_load(file)
        self.thresholds = config['patterns']['matching_threshold']
        self._warned_patterns = set()
    
    def match(self, embedding: np.ndarray, class_name: str) -> Tuple[Optional[str], float]:
        """
        Dopasowanie osadzenia do wzorca.
        
        Args:
            embedding, class_name
            
        Returns:
            (nazwa_dopasowania, podobieństwo)
        """
        patterns = self.database.get_patterns_klasa(class_name)
        if not patterns:
            logger.debug(f"Brak wzorców dla klasy: {class_name}")
            return None, 0.0
        
        # Wybierz próg dla danej klasy
        threshold = self.thresholds.get(class_name, self.thresholds['default'])
        
        best_match = None
        best_similarity = 0.0
        
        emb_flat = embedding.flatten()
        for pattern in patterns:
            pat_emb = pattern['embedding'].flatten()
            if emb_flat.shape != pat_emb.shape:
                if pattern['name'] not in self._warned_patterns:
                    logger.warning(
                        f"Niezgodność wymiarów embeddingu: aktualny {emb_flat.shape} vs wzorzec {pat_emb.shape} "
                        f"dla wzorca '{pattern['name']}' ({class_name}). Pomijanie dopasowania."
                    )
                    self._warned_patterns.add(pattern['name'])
                continue

            similarity = 1 - cosine(emb_flat, pat_emb)
            
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