"""Moduł obsługi kamery."""
import cv2
import numpy as np
import yaml
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class CameraManager:
    """Zarządza połączeniem z kamerą."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.config = config['camera']
        self.cap = None
    
    def open(self) -> bool:
        """Otwórz połączenie z kamerą."""
        backend = self.config.get('backend', 'V4L2')
        
        if backend == 'V4L2':
            self.cap = cv2.VideoCapture(self.config['device_id'], cv2.CAP_V4L2)
        elif backend == 'GSTREAMER':
            # Dla Jetson - użyj GStreamer pipeline
            pipeline = (
                f"v4l2src device=/dev/video{self.config['device_id']} ! "
                "video/x-raw, width=640, height=480 ! "
                "nvvidconv ! video/x-raw, format=BGRx ! "
                "videoconvert ! video/x-raw, format=BGR ! appsink"
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(self.config['device_id'])
        
        if not self.cap.isOpened():
            logger.error(f"Nie można otworzyć kamery {self.config['device_id']}")
            return False
        
        # Ustaw rozdzielczość
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
        
        logger.info(f"Kamera otwarta: {self.config['width']}x{self.config['height']}")
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Odczytaj klatkę z kamery."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def capture_photo(self, window_name: str = "Kamera", 
                     prompt: str = "SPACJA - zdjęcie, ESC - anuluj") -> Optional[np.ndarray]:
        """
        Przechwyć pojedyncze zdjęcie.
        
        Args:
            window_name: Nazwa okna
            prompt: Tekst instrukcji
            
        Returns:
            Zdjęcie lub None
        """
        if not self.open():
            return None
        
        captured = None
        
        while True:
            ret, frame = self.read_frame()
            if not ret:
                break
            
            # Wyświetl instrukcję
            display = frame.copy()
            cv2.putText(display, prompt, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(30) & 0xFF
            if key == 32:  # SPACJA
                captured = frame.copy()
                break
            elif key == 27:  # ESC
                break
        
        self.close()
        cv2.destroyWindow(window_name)
        return captured
    
    def close(self):
        """Zamknij połączenie z kamerą."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Kamera zamknięta")