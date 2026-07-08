"""Moduł ekstrakcji cech wizualnych."""
import cv2, torch
import numpy as np
from torchvision import transforms
import torchvision.models as tvmodels

from typing import Optional
import logging, yaml

logger = logging.getLogger(__name__) #nazwa modułu - do debugu

class FeatureExtractor:
    """Ekstraktor cech wizualnych (EfficientNet + opcjonalnie OSNet)."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as file:
            self.config = yaml.safe_load(file)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_half = self.config['system']['use_half_precision'] and self.device.type == 'cuda'
        
        # EfficientNet - główny ekstraktor
        self._init_efficientnet()
        
        # OSNet dla osób - opcjonalnie
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
                               std=[0.229, 0.224, 0.225])])
            # Sztywna normalizacja dla EfficientNet - wartości działają na większość modeli sieci
        logger.info(f"Zainicjalizowano {model_name}") #Log o inicjalizacji modelu sieci EN
    
    def _init_osnet(self):
        """Inicjalizacja OSNet
        obsługa Reid za pomocą torchreid"""
        try:
            import torchreid
            self.osnet = torchreid.models.build_model(
                name='osnet_x0_25', num_classes=1000, 
                pretrained=True).to(self.device).eval()
            
            logger.info("OSNet załadowany pomyślnie") #log o inicjalizacji OSNet

        except ImportError:
            logger.warning("torchreid nie znaleziony - kodowanie osób przez EfficientNet") #torchreid - nie powodzenie
            self.osnet = None
    
    @torch.no_grad()
    def extract(self, image_crop: np.ndarray, class_name: str = "object") -> Optional[np.ndarray]:
        """ Zwraca wektor cech (w formie numpy array) lub None"""
        if image_crop.size == 0:
            return None
        #wykorzystanie OSNet
        if class_name == 'person' and self.osnet is not None:
            return self._extract_osnet(image_crop)
        #ogólne wykorzystanie EfficientNet
        return self._extract_efficientnet(image_crop)
    
    def _extract_osnet(self, crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        features = self.osnet(tensor)
        return features.flatten().cpu().numpy()
    
    def _extract_efficientnet(self, crop: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        if self.use_half:
            tensor = tensor.half()
        
        features = self.effnet(tensor)
        return features.flatten().cpu().numpy()

    #Ekstrakcja cech dla wielu fragmentów.
    def batch_extract(self, crops: list, class_names: list) -> list: 
        return [self.extract(crop, cls) for crop, cls in zip(crops, class_names)]