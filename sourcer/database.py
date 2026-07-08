"""Moduł zarządzania bazą wzorców."""

import numpy as np

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging, yaml, json

logger = logging.getLogger(__name__) #do debugu

class PatternsDatabase:
    """Baza wzorców obiektów."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as file:
            config = yaml.safe_load(file)
        
        self.db_path = Path(config['patterns']['database_path'])
        self.db_path.mkdir(exist_ok=True)
        self.patterns: Dict[str, List[Dict]] = {}
        self._load()
    
    def _load(self):
        """Wczytaj wszystkie wzorce z dysku."""
        meta_file = self.db_path / "metadata.json"
        if not meta_file.exists():
            logger.info("Baza wzorców jest pusta") #informacja do debugu o bazie wzrocow
            return
        
        with open(meta_file) as file:
            metadata = json.load(file)
        
        for class_name, patterns_list in metadata.items():
            self.patterns[class_name] = []
            for pattern_info in patterns_list:
                #emb_file = self.db_path / f"{pattern_info['name']}_{class_name}.npy"
                #usuniecie nazwy klasy - nazwa klasy jest w ['name']
                emb_file = self.db_path / f"{pattern_info['name']}.npy"
                if emb_file.exists():
                    embedding = np.load(emb_file)
                    self.patterns[class_name].append({
                        'name': pattern_info['name'],
                        'embedding': embedding,
                        'confidence': pattern_info.get('confidence', 0.0),
                        'timestamp': pattern_info.get('timestamp', ''),
                        'metadata': pattern_info.get('metadata', {})})
        
        total = sum(len(v) for v in self.patterns.values())
        logger.info(f"Wczytano {total} wzorców dla {len(self.patterns)} klas")
    
    def _save(self):
        """Zapisz metadane na dysk."""
        metadata = {}
        for class_name, patterns_list in self.patterns.items():
            metadata[class_name] = []
            for pattern in patterns_list:
                # Zapisz embedding
                emb_file = self.db_path / f"{pattern['name']}_{class_name}.npy"
                np.save(emb_file, pattern['embedding'])
                
                # Dodaj do metadanych
                metadata[class_name].append({
                    'name': pattern['name'],
                    'confidence': pattern['confidence'],
                    'timestamp': pattern['timestamp'],
                    'metadata': pattern.get('metadata', {})})
        
        with open(self.db_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def add_pattern(self, name: str, class_name: str, 
                   embedding: np.ndarray, confidence: float = 0.0,
                   metadata: Dict = None) -> bool:
        """
        Dodanie nowego wzorca
        
        Args:
            name: Nazwa wzorca
            class_name: Klasa obiektu
            embedding: Wektor cech
            confidence: Pewność detekcji
            metadata: Dodatkowe metadane
            
        Returns:
            True, jeśli dodano pomyślnie
        """
        if class_name not in self.patterns:
            self.patterns[class_name] = []
        
        # Sprawdź czy wzorzec o tej nazwie już istnieje
            #aktualizacja istniejacych
        for pattern in self.patterns[class_name]:
            if pattern['name'] == name:
                pattern['embedding'] = embedding
                pattern['confidence'] = confidence
                pattern['timestamp'] = datetime.now().isoformat()
                pattern['metadata'] = metadata or {}
                self._save()
                logger.info(f"Zaktualizowano wzorzec: {name} ({class_name})") #informacja do debugu o aktualizacji wzorca
                return True
        
        # Dodanie nowego
        self.patterns[class_name].append({
            'name': name,
            'embedding': embedding,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
        
        self._save()
        logger.info(f"Dodano wzorzec: {name} ({class_name})") #Informacja dodania wzorca
        return True
    
    def remove_pattern(self, name: str, class_name: str = None) -> bool:
        classes = [class_name] if class_name else list(self.patterns.keys())
        removed = False
        
        for klasa in classes:
            if klasa in self.patterns:
                original_len = len(self.patterns[klasa])
                self.patterns[klasa] = [pattern for pattern in self.patterns[klasa] if pattern['name'] != name]
                if len(self.patterns[klasa]) < original_len:
                    emb_file = self.db_path / f"{name}_{klasa}.npy"
                    if emb_file.exists():
                        emb_file.unlink()
                    removed = True

                    # Usuń klasę jeśli pusta
                    if not self.patterns[klasa]:
                        del self.patterns[klasa]
        if removed:
            self._save()
            logger.info(f"Usunięto wzorzec: {name}")
        return removed
    
    #Pobierz wszystkie wzorce dla jednej klasy.
    def get_patterns_klasa(self, class_name: str) -> List[Dict]:
        return self.patterns.get(class_name, [])
    
    def get_all_patterns(self) -> List[Dict]:
        all_patterns = []
        for class_name, patterns_list in self.patterns.items():
            for pattern in patterns_list:
                all_patterns.append({
                    'name': pattern['name'],
                    'class': class_name,
                    'confidence': pattern['confidence'],
                    'timestamp': pattern['timestamp'],
                    'metadata': pattern.get('metadata', {})
                })
        return sorted(all_patterns, key=lambda x: (x['class'], x['name']))
    
    def get_statistics(self) -> Dict:
        total = sum(len(v) for v in self.patterns.values())
        return {
            'total_patterns': total,
            'total_classes': len(self.patterns),
            'classes': {k: len(v) for k, v in self.patterns.items()}}
    
    def clear(self):
        """Usuń wszystkie wzorce."""
        self.patterns.clear()
        # Usuń wszystkie pliki
        for file in self.db_path.glob("*"):
            file.unlink()
        logger.info("Baza wzorców wyczyszczona") #informacja o wyczyszczeniu wszystkich wzorcow