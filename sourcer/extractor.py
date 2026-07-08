"""Moduł ekstrakcji cech wizualnych."""
import cv2
import torch
import numpy as np
import yaml
from torchvision import transforms
import torchvision.models as tvmodels
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Ekstraktor cech wizualnych (EfficientNet + opcjonalnie OSNet)."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_half = self.config['system']['use_half_precision'] and self.device.type == 'cuda'
        
        # EfficientNet jako główny ekstraktor
        self._init_efficientnet()
        
        # OSNet dla osób (opcjonalnie)
        self.osnet = None
        if self.config['models']['feature_extractor']['use_osnet_for_persons']:
            self._init_osnet()
    
    def _init_efficientnet(self):
        """Inicjalizacja EfficientNet."""
        model_name = self.config['models']['feature_extractor']['model_name']
        weights = getattr(tvmodels, f'EfficientNet_B0_Weights').IMAGENET1K_V1
        base = tvmodels.efficientnet_b0(weights=weights)
        self.effnet = torch.nn.Sequential(*list(base.children())[:-1])
        self.effnet.to(self.device).eval()
        
        if self.use_half:
            self.effnet.half()
            logger.info("EfficientNet w trybie half precision")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Zainicjalizowano {model_name}")
    
    def _init_osnet(self):
        """Inicjalizacja OSNet dla lepszego rozpoznawania osób."""
        try:
            import torchreid
            self.osnet = torchreid.models.build_model(
                name='osnet_x0_25', 
                num_classes=1000, 
                pretrained=True
            ).to(self.device).eval()
            logger.info("OSNet załadowany pomyślnie")
        except ImportError:
            logger.warning("torchreid nie znaleziony - osoby będą kodowane EfficientNet")
            self.osnet = None
    
    @torch.no_grad()
    def extract(self, image_crop: np.ndarray, class_name: str = "object") -> Optional[np.ndarray]:
        """
        Wyekstrahuj cechy z fragmentu obrazu.
        
        Args:
            image_crop: Fragment obrazu (BGR)
            class_name: Nazwa klasy obiektu
            
        Returns:
            Wektor cech (numpy array) lub None
        """
        if image_crop.size == 0:
            return None
        
        # Użyj OSNet dla osób
        if class_name == 'person' and self.osnet is not None:
            return self._extract_osnet(image_crop)
        
        # EfficientNet dla pozostałych
        return self._extract_efficientnet(image_crop)
    
    def _extract_osnet(self, crop: np.ndarray) -> np.ndarray:
        """Ekstrakcja cech za pomocą OSNet."""
        img = cv2.resize(crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        features = self.osnet(tensor)
        return features.flatten().cpu().numpy()
    
    def _extract_efficientnet(self, crop: np.ndarray) -> np.ndarray:
        """Ekstrakcja cech za pomocą EfficientNet."""
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        if self.use_half:
            tensor = tensor.half()
        
        features = self.effnet(tensor)
        return features.flatten().cpu().numpy()
    
    def batch_extract(self, crops: list, class_names: list) -> list:
        """Ekstrakcja cech dla wielu fragmentów."""
        return [self.extract(crop, cls) for crop, cls in zip(crops, class_names)]