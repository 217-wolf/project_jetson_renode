#!/usr/bin/env python3
"""
Rozszerzony system rozpoznawania obiektów - wykrywanie wszystkich klas
i porównywanie z bazą wzorców
"""

import cv2
import numpy as np
import torch
import sys
import os
import json 
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ===================== KONFIGURACJA =====================
CAMERA_ID = 0
USE_CUDA = True
PATTERNS_DIR = "patterns_database"
# ==========================================================

# Utwórz katalog na wzorce
Path(PATTERNS_DIR).mkdir(exist_ok=True)

# ---------- Inicjalizacja YOLO ----------
model_yolo = YOLO('yolov8n.pt')
if USE_CUDA and torch.cuda.is_available():
    model_yolo.to('cuda')
    print("YOLO na GPU")
else:
    print("YOLO na CPU")

# Lista klas COCO
COCO_CLASSES = sorted(model_yolo.names.values())

# ---------- Modele embeddingowe ----------
device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')

def build_osnet():
    try:
        import torchreid
        model = torchreid.models.build_model(
            name='osnet_x0_25', num_classes=1000, pretrained=True
        ).to(device).eval()
        return model
    except ImportError:
        return None

def build_efficientnet():
    import torchvision.models as models
    efficientnet = models.efficientnet_b0(pretrained=True)
    model = torch.nn.Sequential(*list(efficientnet.children())[:-1]).to(device).eval()
    return model

osnet_model = build_osnet()
effnet_model = build_efficientnet()
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

def get_embedding_crop(crop, class_name):
    """Helper do generowania embeddingu (używany przez kamerę)."""
    if class_name == 'person' and osnet_model is not None:
        img = cv2.resize(crop, (128,256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
        tensor = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0).to(device)
        with torch.no_grad():
            feat = osnet_model(tensor)
    else:
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = preprocess(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = effnet_model(tensor)
    return feat.flatten().cpu().numpy()

class SingleImageInstanceAnalyzer:
    """Analizuje pojedynczy obraz – wykrywa obiekty wszystkich klas i
    nadaje unikalne ID egzemplarzom tej samej klasy (grupy wizualne)."""

    def __init__(self, model_yolo, device, osnet_model, effnet_model):
        self.detector = model_yolo
        self.device = device
        self.osnet = osnet_model
        self.effnet = effnet_model

    def get_embedding(self, crop, class_name):
        """Tak samo jak wcześniej – zwraca embedding dla wycinka."""
        if crop.size == 0:
            return None
        if class_name == 'person' and self.osnet is not None:
            img = cv2.resize(crop, (128, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img_tensor = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.osnet(img_tensor)
        else:
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_tensor = preprocess(img_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.effnet(img_tensor)
        return feat.flatten().cpu().numpy()

    def process_image(self, image_path):
        """Główna funkcja – zwraca obraz z adnotacjami i słownik detekcji."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Nie można wczytać: {image_path}")

        # Wykryj wszystkie obiekty (bez filtrowania klas)
        results = self.detector(img, conf=0.25, imgsz=640, verbose=False)
        detections = []   # lista: {bbox, class, confidence, embedding}

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    cls_name = self.detector.names[cls_id]

                    crop = img[y1:y2, x1:x2]
                    emb = self.get_embedding(crop, cls_name)
                    if emb is not None:
                        detections.append({
                            'bbox': (x1,y1,x2,y2),
                            'class': cls_name,
                            'confidence': conf,
                            'embedding': emb
                        })

        # Grupowanie w obrębie każdej klasy
        class_groups = self._cluster_instances(detections)

        # Rysowanie ramek
        annotated = img.copy()
        for det in detections:
            x1,y1,x2,y2 = det['bbox']
            cls = det['class']
            conf = det['confidence']
            instance_id = det['instance_id']   # dodane przez _cluster_instances

            color = self._color_for_id(instance_id)
            label = f"{cls}_{instance_id} | {conf:.2f}"

            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1,y1-th-5), (x1+tw,y1), color, -1)
            cv2.putText(annotated, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,255), 2)

        return annotated, detections

    def _cluster_instances(self, detections):
        """Przydziela unikalne ID (np. car_1, car_2) na podstawie podobieństwa wizualnego."""
        # Grupuj detekcje według klasy
        by_class = defaultdict(list)
        for det in detections:
            by_class[det['class']].append(det)

        # Dla każdej klasy wykonaj klasteryzację aglomeracyjną
        for cls_name, items in by_class.items():
            embeddings = np.array([it['embedding'] for it in items])
            n = len(items)
            if n == 0:
                continue
            if n == 1:
                items[0]['instance_id'] = 1
                continue

            # Klasteryzacja (distance_threshold określa maksymalną odległość,
            # przy której obiekty są uznawane za podobne; im mniejsza, tym więcej klastrów)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.1,   # 1 - cosine_similarity -> dystans = 1 - sim
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            # Mapowanie etykiet klastrów na czytelne ID (zaczynając od 1)
            unique_labels = sorted(set(labels))
            label_to_id = {lab: i+1 for i, lab in enumerate(unique_labels)}
            for det, lab in zip(items, labels):
                det['instance_id'] = label_to_id[lab]

        return detections

    def _color_for_id(self, instance_id):
        """Zwraca kolor na podstawie ID (dla lepszej wizualizacji)."""
        # Przesunięcie koloru w HSV
        hue = (instance_id * 43) % 180   # 43° daje różne barwy
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

# ---------- Baza wzorców ----------
class PatternsDatabase:
    def __init__(self):
        self.patterns = {}  # {class_name: [{'name': ..., 'embedding': ..., 'confidence': ...}]}
        self.load_patterns()
    
    def load_patterns(self):
        """Wczytaj wzorce z plików"""
        metadata_file = Path(PATTERNS_DIR) / "patterns.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for class_name, patterns_list in metadata.items():
                self.patterns[class_name] = []
                for pattern_info in patterns_list:
                    emb_file = Path(PATTERNS_DIR) / f"{pattern_info['name']}_{class_name}.npy"
                    if emb_file.exists():
                        embedding = np.load(emb_file)
                        self.patterns[class_name].append({
                            'name': pattern_info['name'],
                            'embedding': embedding,
                            'confidence': pattern_info['confidence'],
                            'timestamp': pattern_info.get('timestamp', '')
                        })
            print(f"Wczytano wzorce dla {len(self.patterns)} klas")
    
    def save_patterns(self):
        """Zapisz wzorce do plików"""
        metadata = {}
        for class_name, patterns_list in self.patterns.items():
            metadata[class_name] = []
            for pattern in patterns_list:
                # Zapisz embedding
                emb_file = Path(PATTERNS_DIR) / f"{pattern['name']}_{class_name}.npy"
                np.save(emb_file, pattern['embedding'])
                # Dodaj do metadanych
                metadata[class_name].append({
                    'name': pattern['name'],
                    'confidence': pattern['confidence'],
                    'timestamp': pattern.get('timestamp', datetime.now().isoformat())
                })
        
        # Zapisz metadane
        with open(Path(PATTERNS_DIR) / "patterns.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def add_pattern(self, name, class_name, embedding, confidence):
        """Dodaj nowy wzorzec"""
        if class_name not in self.patterns:
            self.patterns[class_name] = []
        
        # Sprawdź, czy już istnieje wzorzec o tej nazwie
        existing = [p for p in self.patterns[class_name] if p['name'] == name]
        if existing:
            # Aktualizuj istniejący
            existing[0]['embedding'] = embedding
            existing[0]['confidence'] = confidence
            existing[0]['timestamp'] = datetime.now().isoformat()
        else:
            self.patterns[class_name].append({
                'name': name,
                'embedding': embedding,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
        
        self.save_patterns()
        print(f"Dodano wzorzec '{name}' dla klasy '{class_name}'")
    
    def remove_pattern(self, name, class_name=None):
        """Usuń wzorzec"""
        if class_name:
            classes_to_check = [class_name]
        else:
            classes_to_check = list(self.patterns.keys())
        
        for cls in classes_to_check:
            if cls in self.patterns:
                self.patterns[cls] = [p for p in self.patterns[cls] if p['name'] != name]
                # Usuń plik
                emb_file = Path(PATTERNS_DIR) / f"{name}_{cls}.npy"
                if emb_file.exists():
                    emb_file.unlink()
        
        self.save_patterns()
        print(f"Usunięto wzorzec '{name}'")
    
    def get_patterns_for_class(self, class_name):
        """Pobierz wszystkie wzorce dla danej klasy"""
        return self.patterns.get(class_name, [])
    
    def get_all_classes(self):
        """Pobierz listę klas, dla których mamy wzorce"""
        return list(self.patterns.keys())
    
    def get_all_patterns_list(self):
        """Pobierz listę wszystkich wzorców"""
        all_patterns = []
        for class_name, patterns_list in self.patterns.items():
            for pattern in patterns_list:
                all_patterns.append({
                    'name': pattern['name'],
                    'class': class_name,
                    'confidence': pattern['confidence'],
                    'timestamp': pattern.get('timestamp', '')
                })
        return sorted(all_patterns, key=lambda x: (x['class'], x['name']))

# Inicjalizacja bazy
patterns_db = PatternsDatabase()

# ---------- Funkcje embeddingowe ----------
def get_embedding(image_crop, target_class):
    """Generuj embedding dla obiektu"""
    if image_crop.size == 0:
        return None
    
    if target_class == 'person' and osnet_model is not None:
        # OSNet dla osób
        img = cv2.resize(image_crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
        with torch.no_grad():
            features = osnet_model(img_tensor)
    else:
        # EfficientNet dla pozostałych
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            features = effnet_model(img_tensor)
    
    return features.flatten().cpu().numpy()

def compare_embeddings(emb1, emb2):
    """Porównaj embeddingi (podobieństwo cosinusowe)"""
    return 1 - cosine(emb1.flatten(), emb2.flatten())

# ---------- Funkcje detekcji i rozpoznawania ----------
def detect_all_objects(image):
    """Wykryj WSZYSTKIE obiekty na obrazie"""
    results = model_yolo(image, conf=0.25, imgsz=640, verbose=False)
    objects = []
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                class_name = model_yolo.names[class_id]
                objects.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': class_name,
                    'confidence': conf
                })
    
    return objects

def match_object(embedding, class_name):
    """Dopasuj obiekt do wzorca z bazy"""
    patterns = patterns_db.get_patterns_for_class(class_name)
    
    if not patterns:
        return None, 0.0
    
    threshold = 0.75 if class_name == 'person' else 0.85
    best_match = None
    best_similarity = 0.0
    
    for pattern in patterns:
        similarity = compare_embeddings(embedding, pattern['embedding'])
        if similarity > best_similarity and similarity >= threshold:
            best_similarity = similarity
            best_match = pattern['name']
    
    return best_match, best_similarity

def process_image_full(image_path=None, image=None):
    """Pełne przetwarzanie obrazu - wszystkie klasy, wszystkie wzorce"""
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Nie można wczytać obrazu: {image_path}")
            return None, []
    else:
        img = image
    
    # Wykryj wszystkie obiekty
    objects = detect_all_objects(img)
    annotated = img.copy()
    detections = []
    
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        class_name = obj['class']
        confidence = obj['confidence']
        
        # Wytnij i wygeneruj embedding
        crop = img[y1:y2, x1:x2]
        embedding = get_embedding(crop, class_name)
        
        if embedding is not None:
            # Dopasuj do wzorca
            match_name, similarity = match_object(embedding, class_name)
            
            # Kolory i etykiety
            if match_name:
                color = (0, 255, 0)  # Zielony - rozpoznany
                label = f"{match_name} ({class_name})"
                status = "MATCH"
            else:
                color = (0, 0, 255)  # Czerwony - nierozpoznany
                label = f"UNKNOWN ({class_name})"
                status = "UNKNOWN"
            
            # Rysuj ramkę
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Tekst informacyjny
            text = f"{label} | {confidence:.2f}"
            if match_name:
                text += f" | sim:{similarity:.3f}"
            
            # Tło pod tekst
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1-text_h-5), (x1+text_w, y1), color, -1)
            cv2.putText(annotated, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 2)
            
            # Log
            if match_name:
                print(f"[{class_name}] (pewność {confidence:.2f}) -> "
                      f"dopasowanie: {match_name} (podobieństwo: {similarity:.4f})")
            else:
                print(f"[{class_name}] (pewność {confidence:.2f}) -> NIEZNANY")
            
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'match': match_name or 'UNKNOWN',
                'similarity': similarity
            })
    
    return annotated, detections

# ---------- Funkcje kamery ----------
def capture_from_camera(window_name="Kamera", instruction="Naciśnij SPACJĘ, aby zrobić zdjęcie"):
    """Przechwyć obraz z kamery"""
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Nie można otworzyć kamery.")
        return None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(instruction)
    captured_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        cv2.putText(display, instruction, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 32:  # SPACJA
            captured_frame = frame.copy()
            break
        elif key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyWindow(window_name)
    return captured_frame

def run_camera_test():
    """Uruchom test na żywo z kamery"""
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Nie można otworzyć kamery.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n=== Test z kamery (wszystkie klasy) ===")
    print("Naciśnij 'q', aby zakończyć")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Przetwórz klatkę
        annotated, detections = process_image_full(image=frame)
        
        # Wyświetl
        cv2.imshow("Test z kamery (wszystkie klasy)", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyWindow("Test z kamery (wszystkie klasy)")

# ---------- GUI Tkinter ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("System rozpoznawania obiektów - Multi-Class")
        root.geometry("800x600")
        
        # Sprawdź modele
        self.osnet_available = osnet_model is not None
        
        # Konfiguracja interfejsu
        self.setup_ui()
        self.refresh_patterns_list()
    
    def setup_ui(self):
        """Konfiguracja interfejsu użytkownika"""
        # Notebook z zakładkami
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Zakładka 1: Dodawanie wzorców
        add_frame = ttk.Frame(notebook)
        notebook.add(add_frame, text="Dodaj wzorce")
        self.setup_add_tab(add_frame)
        
        # Zakładka 2: Test
        test_frame = ttk.Frame(notebook)
        notebook.add(test_frame, text="Test")
        self.setup_test_tab(test_frame)
        
        # Zakładka 3: Lista wzorców
        list_frame = ttk.Frame(notebook)
        notebook.add(list_frame, text="Lista wzorców")
        self.setup_list_tab(list_frame)
        quick_frame = ttk.Frame(notebook)
        notebook.add(quick_frame, text="Szybka analiza")
        self.setup_quick_tab(quick_frame)
        # Status bar
        self.status_var = tk.StringVar(value="Gotowy. Wybierz zakładkę.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
    
    def setup_add_tab(self, parent):
        """Zakładka dodawania wzorców"""
        # Ramka dla pojedynczego wzorca
        single_frame = ttk.LabelFrame(parent, text="Dodaj pojedynczy wzorzec")
        single_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Nazwa wzorca
        ttk.Label(single_frame, text="Nazwa wzorca:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.pattern_name_var = tk.StringVar()
        ttk.Entry(single_frame, textvariable=self.pattern_name_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        
        # Klasa
        ttk.Label(single_frame, text="Klasa obiektu:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.pattern_class_var = tk.StringVar()
        class_combo = ttk.Combobox(single_frame, textvariable=self.pattern_class_var, 
                                   values=COCO_CLASSES, state="readonly", width=27)
        class_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Źródło
        ttk.Label(single_frame, text="Źródło:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.add_source_var = tk.StringVar(value="file")
        ttk.Radiobutton(single_frame, text="Z pliku", variable=self.add_source_var, 
                       value="file").grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Radiobutton(single_frame, text="Z kamery", variable=self.add_source_var, 
                       value="camera").grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Przycisk dodawania
        ttk.Button(single_frame, text="DODAJ WZORZEC", 
                  command=self.add_single_pattern).grid(row=4, column=0, columnspan=2, pady=15)
        
        # Ramka dla wzorców z obrazu
        multi_frame = ttk.LabelFrame(parent, text="Dodaj wszystkie obiekty z obrazu jako wzorce")
        multi_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(multi_frame, text="Wszystkie wykryte obiekty zostaną dodane jako wzorce\n"
                                   "z nazwami: nazwa_klasy_001, nazwa_klasy_002, itd.").pack(pady=10)
        
        ttk.Button(multi_frame, text="DODAJ WSZYSTKIE OBIEKTY Z PLIKU", 
                  command=self.add_all_from_file).pack(pady=10)
        ttk.Button(multi_frame, text="DODAJ WSZYSTKIE OBIEKTY Z KAMERY", 
                  command=self.add_all_from_camera).pack(pady=5)
    
    def setup_test_tab(self, parent):
        """Zakładka testowa"""
        # Wybór źródła
        source_frame = ttk.LabelFrame(parent, text="Wybierz źródło testu")
        source_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.test_source_var = tk.StringVar(value="file")
        ttk.Radiobutton(source_frame, text="Z pliku", variable=self.test_source_var, 
                       value="file").pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Radiobutton(source_frame, text="Z kamery", variable=self.test_source_var, 
                       value="camera").pack(side=tk.LEFT, padx=10, pady=10)
        
        # Przycisk START
        ttk.Button(source_frame, text="ROZPOCZNIJ TEST", 
                  command=self.start_test).pack(side=tk.LEFT, padx=20, pady=10)
        
        # Panel wyników
        results_frame = ttk.LabelFrame(parent, text="Wyniki ostatniego testu")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_text = tk.Text(results_frame, height=15, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_list_tab(self, parent):
        """Zakładka z listą wzorców"""
        # Przyciski
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="Odśwież listę", 
                  command=self.refresh_patterns_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Usuń zaznaczony", 
                  command=self.remove_selected_pattern).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Usuń wszystkie", 
                  command=self.remove_all_patterns).pack(side=tk.LEFT, padx=5)
        
        # Lista wzorców
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview dla lepszego wyświetlania
        columns = ('name', 'class', 'confidence', 'timestamp')
        self.patterns_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        self.patterns_tree.heading('name', text='Nazwa wzorca')
        self.patterns_tree.heading('class', text='Klasa')
        self.patterns_tree.heading('confidence', text='Pewność')
        self.patterns_tree.heading('timestamp', text='Data dodania')
        
        self.patterns_tree.column('name', width=200)
        self.patterns_tree.column('class', width=150)
        self.patterns_tree.column('confidence', width=100)
        self.patterns_tree.column('timestamp', width=150)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.patterns_tree.yview)
        self.patterns_tree.configure(yscroll=scrollbar.set)
        
        self.patterns_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def add_single_pattern(self):
        """Dodaj pojedynczy wzorzec"""
        name = self.pattern_name_var.get().strip()
        class_name = self.pattern_class_var.get().strip()
        source = self.add_source_var.get()
        
        if not name or not class_name:
            messagebox.showerror("Błąd", "Uzupełnij nazwę i klasę wzorca!")
            return
        
        # Pobierz obraz
        if source == "file":
            filepath = filedialog.askopenfilename(
                title="Wybierz obraz wzorca",
                filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")]
            )
            if not filepath:
                return
            image = cv2.imread(filepath)
        else:
            self.root.withdraw()
            image = capture_from_camera("Dodawanie wzorca - kamera")
            self.root.deiconify()
        
        if image is None:
            return
        
        # Wykryj obiekt
        class_id = list(model_yolo.names.keys())[list(model_yolo.names.values()).index(class_name)]
        results = model_yolo(image, classes=[class_id], conf=0.25, verbose=False)
        
        detected = False
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                box = result.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0].item())
                
                crop = image[y1:y2, x1:x2]
                embedding = get_embedding(crop, class_name)
                
                if embedding is not None:
                    patterns_db.add_pattern(name, class_name, embedding, confidence)
                    messagebox.showinfo("Sukces", f"Dodano wzorzec '{name}' ({class_name})")
                    detected = True
                    break
        
        if not detected:
            messagebox.showerror("Błąd", f"Nie wykryto obiektu klasy '{class_name}'")
        
        self.refresh_patterns_list()
    
    def add_all_from_file(self):
        """Dodaj wszystkie obiekty z pliku jako wzorce"""
        filepath = filedialog.askopenfilename(
            title="Wybierz obraz z obiektami",
            filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filepath:
            return
        
        image = cv2.imread(filepath)
        if image is None:
            messagebox.showerror("Błąd", "Nie można wczytać obrazu")
            return
        
        objects = detect_all_objects(image)
        added_count = 0
        
        for i, obj in enumerate(objects):
            x1, y1, x2, y2 = obj['bbox']
            class_name = obj['class']
            confidence = obj['confidence']
            
            crop = image[y1:y2, x1:x2]
            embedding = get_embedding(crop, class_name)
            
            if embedding is not None:
                # Generuj nazwę
                name = f"{class_name}_{i+1:03d}"
                patterns_db.add_pattern(name, class_name, embedding, confidence)
                added_count += 1
        
        messagebox.showinfo("Sukces", f"Dodano {added_count} wzorców z obrazu")
        self.refresh_patterns_list()
    
    def add_all_from_camera(self):
        """Dodaj wszystkie obiekty z kamery jako wzorce"""
        self.root.withdraw()
        image = capture_from_camera("Dodawanie wzorców - kamera")
        self.root.deiconify()
        
        if image is None:
            return
        
        objects = detect_all_objects(image)
        added_count = 0
        
        for i, obj in enumerate(objects):
            x1, y1, x2, y2 = obj['bbox']
            class_name = obj['class']
            confidence = obj['confidence']
            
            crop = image[y1:y2, x1:x2]
            embedding = get_embedding(crop, class_name)
            
            if embedding is not None:
                name = f"{class_name}_{i+1:03d}"
                patterns_db.add_pattern(name, class_name, embedding, confidence)
                added_count += 1
        
        messagebox.showinfo("Sukces", f"Dodano {added_count} wzorców z kamery")
        self.refresh_patterns_list()
    
    def start_test(self):
        """Rozpocznij test"""
        test_source = self.test_source_var.get()
        
        print("\n" + "="*60)
        print("ROZPOCZYNANIE TESTU (wszystkie klasy)")
        print("="*60)
        
        self.results_text.delete(1.0, tk.END)
        
        if test_source == "file":
            filepath = filedialog.askopenfilename(
                title="Wybierz obraz testowy",
                filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")]
            )
            if not filepath:
                return
            
            image = cv2.imread(filepath)
            if image is not None:
                annotated, detections = process_image_full(image_path=filepath)
                
                # Zapisz wynik
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"wynik_test_wszystkie_{timestamp}.jpg"
                cv2.imwrite(output_path, annotated)
                
                # Wyświetl wyniki
                self.display_results(detections, output_path)
                cv2.imshow("Wynik testu (wszystkie klasy)", annotated)
                cv2.waitKey(0)
                cv2.destroyWindow("Wynik testu (wszystkie klasy)")
        else:
            # Test z kamery
            self.root.withdraw()
            run_camera_test()
            self.root.deiconify()
    
    def display_results(self, detections, output_path):
        """Wyświetl wyniki w interfejsie"""
        self.results_text.insert(tk.END, f"Zapisano obraz: {output_path}\n")
        self.results_text.insert(tk.END, "="*60 + "\n")
        
        # Grupuj po klasach
        classes_found = {}
        for det in detections:
            cls = det['class']
            if cls not in classes_found:
                classes_found[cls] = []
            classes_found[cls].append(det)
        
        for class_name, class_detections in sorted(classes_found.items()):
            self.results_text.insert(tk.END, f"\n{'-'*40}\n")
            self.results_text.insert(tk.END, f"Klasa: {class_name.upper()}\n")
            self.results_text.insert(tk.END, f"{'-'*40}\n")
            
            for det in class_detections:
                status = "✓" if det['match'] != 'UNKNOWN' else "✗"
                text = f"{status} {det['match']} (pewność: {det['confidence']:.2f}"
                if det['similarity'] > 0:
                    text += f", podobieństwo: {det['similarity']:.4f}"
                text += ")\n"
                self.results_text.insert(tk.END, text)
        
        total = len(detections)
        matched = len([d for d in detections if d['match'] != 'UNKNOWN'])
        self.results_text.insert(tk.END, f"\n{'='*60}\n")
        self.results_text.insert(tk.END, f"RAZEM: {total} obiektów, {matched} rozpoznanych\n")
        
        self.status_var.set(f"Test zakończony. Wykryto {total} obiektów.")
    
    def refresh_patterns_list(self):
        """Odśwież listę wzorców"""
        if hasattr(self, 'patterns_tree'):
            # Wyczyść
            for item in self.patterns_tree.get_children():
                self.patterns_tree.delete(item)
            
            # Dodaj wzorce
            all_patterns = patterns_db.get_all_patterns_list()
            for pattern in all_patterns:
                self.patterns_tree.insert('', tk.END, values=(
                    pattern['name'],
                    pattern['class'],
                    f"{pattern['confidence']:.2f}",
                    pattern['timestamp'][:19] if pattern['timestamp'] else ''
                ))
    
    def remove_selected_pattern(self):
        """Usuń zaznaczony wzorzec"""
        selection = self.patterns_tree.selection()
        if not selection:
            messagebox.showwarning("Uwaga", "Wybierz wzorzec do usunięcia")
            return
        
        item = self.patterns_tree.item(selection[0])
        name = item['values'][0]
        class_name = item['values'][1]
        
        if messagebox.askyesno("Potwierdzenie", f"Czy usunąć wzorzec '{name}' ({class_name})?"):
            patterns_db.remove_pattern(name, class_name)
            self.refresh_patterns_list()
    
    def remove_all_patterns(self):
        """Usuń wszystkie wzorce"""
        if messagebox.askyesno("Potwierdzenie", "Czy na pewno usunąć WSZYSTKIE wzorce?"):
            for class_name in list(patterns_db.patterns.keys()):
                for pattern in patterns_db.patterns[class_name][:]:
                    patterns_db.remove_pattern(pattern['name'], class_name)
            self.refresh_patterns_list()
            messagebox.showinfo("Sukces", "Usunięto wszystkie wzorce")
    def setup_quick_tab(self, parent):
        """Zakładka 'Szybka analiza' – jedno zdjęcie, wszystkie klasy, unikalne ID."""
        frame = ttk.LabelFrame(parent, text="Analiza pojedynczego obrazu")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Wybierz zdjęcie – system wykryje wszystkie obiekty\n"
                            "i nada im unikalne identyfikatory (car_1, car_2 itd.).",
                justify=tk.LEFT).pack(pady=10)

        ttk.Button(frame, text="WYBIERZ ZDJĘCIE I ANALIZUJ",
                command=self.quick_analysis).pack(pady=20)

        self.quick_result_text = tk.Text(frame, height=10, width=80)
        self.quick_result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def quick_analysis(self):
        filepath = filedialog.askopenfilename(
            title="Wybierz zdjęcie do analizy",
            filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filepath:
            return

        # Użyj analizatora
        analyzer = SingleImageInstanceAnalyzer(
            model_yolo, device, osnet_model, effnet_model
        )
        try:
            annotated, detections = analyzer.process_image(filepath)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
            return

        # Zapisz wynik
        out_path = f"analiza_{Path(filepath).stem}.jpg"
        cv2.imwrite(out_path, annotated)
        cv2.imshow("Szybka analiza – unikalne ID", annotated)

        # Pokaż podsumowanie w polu tekstowym
        self.quick_result_text.delete(1.0, tk.END)
        summary = defaultdict(list)
        for det in detections:
            summary[det['class']].append(det['instance_id'])
        for cls, ids in sorted(summary.items()):
            unique_ids = sorted(set(ids))
            self.quick_result_text.insert(tk.END,
                f"{cls}: {len(ids)} obiektów (ID: {', '.join(map(str, unique_ids))})\n")
        self.quick_result_text.insert(tk.END, f"\nWynik zapisano jako: {out_path}\n")
# ---------- Start ----------
def main():
    print("="*60)
    print("ROZSZERZONY SYSTEM ROZPOZNAWANIA OBIEKTÓW")
    print("Wykrywanie wszystkich klas + baza wzorców")
    print("="*60)
    
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()