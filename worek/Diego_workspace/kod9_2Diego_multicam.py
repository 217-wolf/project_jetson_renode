#!/usr/bin/env python3
"""
System rozpoznawania obiektów (YOLOv8 + EfficientNet) – baza wzorców, analiza pojedynczego obrazu.
Zoptymalizowany dla NVIDIA Jetson Orin Nano (TensorRT, half precision).
Obsługa wielu kamer USB w trybie testowym.
"""

import cv2, torch, json, sys, os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
from ultralytics import YOLO
from torchvision import transforms
import torchvision.models as tvmodels
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -------------------------- KONFIGURACJA ----------------------------------------
CAMERA_ID = 0                      # ID domyślnej kamery
USE_CUDA = torch.cuda.is_available()
PATTERNS_DIR = "patterns_database"
Path(PATTERNS_DIR).mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# -------------------------- MODELE ----------------------------------------------
MODEL_PATH = 'yolov8n.engine' if os.path.exists('yolov8n.engine') else 'yolov8n.pt'
model_yolo = YOLO(MODEL_PATH)
if USE_CUDA:
    model_yolo.to('cuda')
print(f"YOLO: {MODEL_PATH} na {'GPU' if USE_CUDA else 'CPU'}")

COCO_CLASSES = sorted(model_yolo.names.values())
NAME2ID = {v: k for k, v in model_yolo.names.items()}

# -------------------------- EKSTRAKTOR CECH (EfficientNet) --------------------
class FeatureExtractor:
    def __init__(self):
        base = tvmodels.efficientnet_b0(weights='IMAGENET1K_V1')
        self.model = torch.nn.Sequential(*list(base.children())[:-1])
        self.model.to(DEVICE).eval()
        if USE_CUDA:
            self.model.half()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        self.osnet = None
        try:
            import torchreid
            self.osnet = torchreid.models.build_model(
                name='osnet_x0_25', num_classes=1000, pretrained=True
            ).to(DEVICE).eval()
            print("OSNet załadowany.")
        except ImportError:
            print("torchreid nie znaleziony.")

    @torch.no_grad()
    def get_embedding(self, crop, class_name):
        if crop.size == 0:
            return None
        if class_name == 'person' and self.osnet is not None:
            img = cv2.resize(crop, (128, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            tensor = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0).to(DEVICE)
            features = self.osnet(tensor)
        else:
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img_rgb).unsqueeze(0).to(DEVICE)
            if USE_CUDA:
                tensor = tensor.half()
            features = self.model(tensor)
        return features.flatten().cpu().numpy()

feature_extractor = FeatureExtractor()

# -------------------------- BAZA WZORCÓW ---------------------------------------
class PatternsDatabase:
    
    def __init__(self):
        self.patterns = {}
        self._load()

    def _load(self):
        meta = Path(PATTERNS_DIR) / "patterns.json"
        if meta.exists():
            with open(meta) as f:
                data = json.load(f)
            for cls, lst in data.items():
                self.patterns[cls] = []
                for p in lst:
                    emb_path = Path(PATTERNS_DIR) / f"{p['name']}_{cls}.npy"
                    if emb_path.exists():
                        emb = np.load(emb_path)
                        self.patterns[cls].append({
                            'name': p['name'],
                            'embedding': emb,
                            'confidence': p['confidence'],
                            'timestamp': p.get('timestamp', '')
                        })
            print(f"Wczytano wzorce dla {len(self.patterns)} klas.")

    def _save(self):
        meta = {}
        for cls, lst in self.patterns.items():
            meta[cls] = []
            for p in lst:
                emb_path = Path(PATTERNS_DIR) / f"{p['name']}_{cls}.npy"
                np.save(emb_path, p['embedding'])
                meta[cls].append({
                    'name': p['name'],
                    'confidence': p['confidence'],
                    'timestamp': p.get('timestamp', datetime.now().isoformat())
                })
        with open(Path(PATTERNS_DIR) / "patterns.json", 'w') as f:
            json.dump(meta, f, indent=2)

    def add(self, name, class_name, embedding, confidence):
        if class_name not in self.patterns:
            self.patterns[class_name] = []
        existing = [p for p in self.patterns[class_name] if p['name'] == name]
        if existing:
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
        self._save()
        print(f"Dodano wzorzec '{name}' ({class_name}).")

    def remove(self, name, class_name=None):
        classes = [class_name] if class_name else list(self.patterns.keys())
        for cls in classes:
            if cls in self.patterns:
                self.patterns[cls] = [p for p in self.patterns[cls] if p['name'] != name]
                (Path(PATTERNS_DIR) / f"{name}_{cls}.npy").unlink(missing_ok=True)
        self._save()

    def get_patterns_for_class(self, class_name):
        return self.patterns.get(class_name, [])

    def get_all_classes(self):
        return list(self.patterns.keys())

    def get_all_patterns_list(self):
        all_p = []
        for cls, lst in self.patterns.items():
            for p in lst:
                all_p.append({
                    'name': p['name'], 'class': cls,
                    'confidence': p['confidence'],
                    'timestamp': p.get('timestamp', '')
                })
        return sorted(all_p, key=lambda x: (x['class'], x['name']))

patterns_db = PatternsDatabase()

# -------------------------- DETEKCJA I DOPASOWANIE ----------------------------
def detect_objects(image, classes=None):
    results = model_yolo(image, classes=classes, conf=0.25, imgsz=640, verbose=False)
    objects = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model_yolo.names[cls_id]
                objects.append({'bbox': (x1,y1,x2,y2), 'class': cls_name, 'confidence': conf})
    return objects

def match_object(embedding, class_name):
    patterns = patterns_db.get_patterns_for_class(class_name)
    if not patterns:
        return None, 0.0
    threshold = 0.75 if class_name == 'person' else 0.85
    best_match, best_sim = None, 0.0
    for p in patterns:
        sim = 1 - cosine(embedding.flatten(), p['embedding'].flatten())
        if sim > best_sim and sim >= threshold:
            best_sim = sim
            best_match = p['name']
    return best_match, best_sim

class SingleImageInstanceAnalyzer:
    
    def __init__(self, detector, device, feature_extractor):
        self.detector = detector
        self.device = device
        self.fe = feature_extractor

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Nie można wczytać: {image_path}")
        detections = []
        results = self.detector(img, conf=0.25, imgsz=640, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.detector.names[cls_id]
                    crop = img[y1:y2, x1:x2]
                    emb = self.fe.get_embedding(crop, cls_name)
                    if emb is not None:
                        detections.append({'bbox': (x1,y1,x2,y2), 'class': cls_name,
                                          'confidence': conf, 'embedding': emb})
        by_class = defaultdict(list)
        for d in detections:
            by_class[d['class']].append(d)
        for cls, items in by_class.items():
            embs = np.array([it['embedding'] for it in items])
            n = len(items)
            if n == 1:
                items[0]['instance_id'] = 1
            elif n > 1:
                clust = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=0.1, metric='cosine', linkage='average')
                labels = clust.fit_predict(embs)
                unique = sorted(set(labels))
                label2id = {l: i+1 for i, l in enumerate(unique)}
                for it, lab in zip(items, labels):
                    it['instance_id'] = label2id[lab]
        annotated = img.copy()
        for d in detections:
            x1,y1,x2,y2 = d['bbox']
            color = self._color_for_id(d['instance_id'])
            label = f"{d['class']}_{d['instance_id']} | {d['confidence']:.2f}"
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1,y1-th-5), (x1+tw,y1), color, -1)
            cv2.putText(annotated, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        return annotated, detections

    @staticmethod
    def _color_for_id(instance_id):
        hue = (instance_id * 43) % 180
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

def process_image_full(image_path=None, image=None):
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            return None, []
    else:
        img = image
    objs = detect_objects(img)
    annotated = img.copy()
    detections = []
    for obj in objs:
        x1,y1,x2,y2 = obj['bbox']
        cls_name = obj['class']
        conf = obj['confidence']
        crop = img[y1:y2, x1:x2]
        emb = feature_extractor.get_embedding(crop, cls_name)
        if emb is not None:
            match_name, sim = match_object(emb, cls_name)
            color = (0,255,0) if match_name else (0,0,255)
            label = f"{match_name} ({cls_name})" if match_name else f"UNKNOWN ({cls_name})"
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            text = f"{label} | {conf:.2f}" + (f" | sim:{sim:.3f}" if match_name else "")
            (tw,th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1,y1-th-5), (x1+tw,y1), color, -1)
            cv2.putText(annotated, text, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            detections.append({'class': cls_name, 'confidence': conf,
                               'bbox': (x1,y1,x2,y2),
                               'match': match_name or 'UNKNOWN', 'similarity': sim})
    return annotated, detections

# -------------------------- KAMERA --------------------------------------------
def capture_from_camera(window_name="Kamera", prompt="Naciśnij SPACJĘ"):
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Nie można otworzyć kamery.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    captured = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        disp = frame.copy()
        cv2.putText(disp, prompt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == 32:
            captured = frame.copy()
            break
        elif key == 27:
            break
    cap.release()
    cv2.destroyWindow(window_name)
    return captured

# -------------------------- TEST WIELU KAMER (POPRAWIONY) ----------------------
def run_multi_camera_test(camera_ids, parent_root=None):
    captures = []
    window_names = []
    failed = []

    for idx in camera_ids:
        print(f"Próba otwarcia kamery {idx}...")
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            failed.append(idx)
            print(f"Kamera {idx}: NIE OTWARTA (sprawdź podłączenie/ID)")
            continue
        # Sprawdźmy, czy odczyt działa
        ret, test_frame = cap.read()
        if not ret:
            print(f"Kamera {idx}: otwarta, ale nie można odczytać klatki. Pomijam.")
            cap.release()
            failed.append(idx)
            continue
        # Ustaw rozdzielczość
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        captures.append(cap)
        win_name = f"Kamera {idx}"
        window_names.append(win_name)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        print(f"Kamera {idx} gotowa – okno '{win_name}'.")

    if failed and parent_root:
        messagebox.showwarning("Ostrzeżenie", f"Nie udało się otworzyć kamer: {failed}")
    if not captures:
        print("Brak działających kamer.")
        if parent_root:
            messagebox.showerror("Błąd", "Nie udało się otworzyć żadnej kamery.")
        return

    print("Test kamer: naciśnij 'q', 'x' lub Esc, aby zakończyć.")
    while True:
        # Przetwarzanie każdej kamery
        for i, cap in enumerate(captures):
            win = window_names[i]
            # Sprawdź, czy okno nadal istnieje (zapobiega odtworzeniu po zamknięciu)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                continue  # nie wyświetlaj, jeśli okno zamknięte
            ret, frame = cap.read()
            if not ret:
                # Jeśli nie można odczytać, zamknij to okno
                cv2.destroyWindow(win)
                continue
            annotated, _ = process_image_full(image=frame)
            cv2.imshow(win, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('x'), 27):
            break

        # Sprawdź, czy wszystkie okna zostały zamknięte
        any_visible = False
        for win in window_names:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1:
                any_visible = True
                break
        if not any_visible:
            print("Wszystkie okna zamknięte – kończę test.")
            break

    # Sprzątanie
    for cap in captures:
        cap.release()
    for win in window_names:
        cv2.destroyWindow(win)  # bezpieczne

    if parent_root:
        messagebox.showinfo("Test zakończony", "Test kamer został przerwany.")
    print("Test zakończony.")

# -------------------------- INTERFEJS GRAFICZNY --------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("System rozpoznawania obiektów – Jetson Orin Nano")
        root.geometry("800x600")
        self.setup_ui()
        self.refresh_patterns_list()

    def setup_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        nb.add(self._create_add_tab(), text="Dodaj wzorce")
        nb.add(self._create_test_tab(), text="Test")
        nb.add(self._create_list_tab(), text="Lista wzorców")
        nb.add(self._create_quick_tab(), text="Szybka analiza")
        self.status_var = tk.StringVar(value="Gotowy.")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, padx=5, pady=5)

    # ---------- DODAWANIE ----------
    def _create_add_tab(self):
        f = ttk.Frame()
        sf = ttk.LabelFrame(f, text="Dodaj pojedynczy wzorzec")
        sf.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(sf, text="Nazwa:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.name_var = tk.StringVar()
        ttk.Entry(sf, textvariable=self.name_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(sf, text="Klasa:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.class_var = tk.StringVar()
        ttk.Combobox(sf, textvariable=self.class_var, values=COCO_CLASSES,
                     state="readonly", width=27).grid(row=1, column=1, padx=5, pady=5)
        self.src_var = tk.StringVar(value="file")
        ttk.Radiobutton(sf, text="Plik", variable=self.src_var, value="file").grid(row=2, column=1, sticky=tk.W)
        ttk.Radiobutton(sf, text="Kamera", variable=self.src_var, value="camera").grid(row=3, column=1, sticky=tk.W)
        ttk.Button(sf, text="DODAJ WZORZEC", command=self._add_single).grid(row=4, column=0, columnspan=2, pady=10)

        mf = ttk.LabelFrame(f, text="Dodaj wszystkie obiekty z obrazu jako wzorce")
        mf.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(mf, text="Wykryte obiekty otrzymają nazwy klasa_001, klasa_002 itd.").pack(pady=5)
        ttk.Button(mf, text="Z PLIKU", command=self._add_all_file).pack(pady=5)
        ttk.Button(mf, text="Z KAMERY", command=self._add_all_camera).pack(pady=5)
        return f

    def _add_single(self):
        name = self.name_var.get().strip()
        cls = self.class_var.get().strip()
        if not name or not cls:
            messagebox.showerror("Błąd", "Podaj nazwę i klasę!")
            return
        if cls not in NAME2ID:
            messagebox.showerror("Błąd", f"Nieznana klasa: {cls}")
            return
        if self.src_var.get() == "file":
            path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")])
            if not path: return
            img = cv2.imread(path)
        else:
            self.root.withdraw()
            img = capture_from_camera("Dodaj wzorzec")
            self.root.deiconify()
        if img is None:
            return
        class_id = NAME2ID[cls]
        objs = detect_objects(img, classes=[class_id])
        if not objs:
            messagebox.showerror("Błąd", f"Nie wykryto obiektu klasy '{cls}'")
            return
        obj = objs[0]
        x1,y1,x2,y2 = obj['bbox']
        crop = img[y1:y2, x1:x2]
        emb = feature_extractor.get_embedding(crop, cls)
        if emb is not None:
            patterns_db.add(name, cls, emb, obj['confidence'])
            messagebox.showinfo("OK", f"Dodano wzorzec '{name}' ({cls})")
        else:
            messagebox.showerror("Błąd", "Nie udało się wygenerować cech.")
        self.refresh_patterns_list()

    def _add_all_file(self):
        path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Błąd", "Nie można wczytać obrazu")
            return
        self._add_all_from_image(img)

    def _add_all_camera(self):
        self.root.withdraw()
        img = capture_from_camera("Dodaj wszystkie obiekty")
        self.root.deiconify()
        if img is not None:
            self._add_all_from_image(img)

    def _add_all_from_image(self, img):
        objs = detect_objects(img)
        cnt = 0
        for i, obj in enumerate(objs):
            cls = obj['class']
            x1,y1,x2,y2 = obj['bbox']
            crop = img[y1:y2, x1:x2]
            emb = feature_extractor.get_embedding(crop, cls)
            if emb is not None:
                name = f"{cls}_{i+1:03d}"
                patterns_db.add(name, cls, emb, obj['confidence'])
                cnt += 1
        messagebox.showinfo("OK", f"Dodano {cnt} wzorców.")
        self.refresh_patterns_list()

    # ---------- TEST ----------
    def _create_test_tab(self):
        f = ttk.Frame()
        sf = ttk.LabelFrame(f, text="Źródło testu")
        sf.pack(fill=tk.X, padx=10, pady=10)
        self.test_src = tk.StringVar(value="file")
        ttk.Radiobutton(sf, text="Plik", variable=self.test_src, value="file").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(sf, text="Kamera", variable=self.test_src, value="camera").pack(side=tk.LEFT, padx=10)

        cam_frame = ttk.Frame(sf)
        cam_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(cam_frame, text="ID kamer (przecinkiem):").pack(side=tk.LEFT)
        self.camera_ids_var = tk.StringVar(value="0")
        ttk.Entry(cam_frame, textvariable=self.camera_ids_var, width=15).pack(side=tk.LEFT, padx=5)

        ttk.Button(sf, text="ROZPOCZNIJ TEST", command=self._start_test).pack(side=tk.LEFT, padx=20)

        rf = ttk.LabelFrame(f, text="Wyniki")
        rf.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.results_text = tk.Text(rf, height=15, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        return f

    def _start_test(self):
        self.results_text.delete(1.0, tk.END)
        if self.test_src.get() == "file":
            path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")])
            if not path: return
            annotated, detections = process_image_full(image_path=path)
            if annotated is not None:
                out = f"test_wynik_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(out, annotated)
                self._display_results(detections, out)
                cv2.imshow("Wynik testu", annotated)
                cv2.waitKey(0)
                cv2.destroyWindow("Wynik testu")
        else:
            # Test z kamer(y)
            ids_str = self.camera_ids_var.get().strip()
            try:
                camera_ids = [int(x.strip()) for x in ids_str.split(',') if x.strip()]
            except ValueError:
                messagebox.showerror("Błąd", "Nieprawidłowe ID kamer. Podaj liczby rozdzielone przecinkami.")
                return
            self.root.withdraw()
            run_multi_camera_test(camera_ids, parent_root=self.root)
            self.root.deiconify()

    def _display_results(self, detections, out_path):
        self.results_text.insert(tk.END, f"Zapisano: {out_path}\n{'='*60}\n")
        classes = defaultdict(list)
        for d in detections:
            classes[d['class']].append(d)
        for cls in sorted(classes):
            self.results_text.insert(tk.END, f"\n--- {cls.upper()} ---\n")
            for d in classes[cls]:
                status = "✓" if d['match'] != 'UNKNOWN' else "✗"
                line = f"{status} {d['match']} (pewność: {d['confidence']:.2f}"
                if d['similarity'] > 0:
                    line += f", podobieństwo: {d['similarity']:.4f}"
                line += ")\n"
                self.results_text.insert(tk.END, line)
        total = len(detections)
        matched = sum(1 for d in detections if d['match'] != 'UNKNOWN')
        self.results_text.insert(tk.END, f"\n{'='*60}\nRAZEM: {total} obiektów, {matched} rozpoznanych.\n")
        self.status_var.set(f"Test zakończony – {matched}/{total} rozpoznanych.")

    # ---------- LISTA WZORCÓW ----------
    def _create_list_tab(self):
        f = ttk.Frame()
        btnf = ttk.Frame(f)
        btnf.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btnf, text="Odśwież", command=self.refresh_patterns_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(btnf, text="Usuń zaznaczony", command=self._remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btnf, text="Usuń wszystkie", command=self._remove_all).pack(side=tk.LEFT, padx=5)

        cols = ('name','class','confidence','timestamp')
        self.tree = ttk.Treeview(f, columns=cols, show='headings')
        self.tree.heading('name', text='Nazwa')
        self.tree.heading('class', text='Klasa')
        self.tree.heading('confidence', text='Pewność')
        self.tree.heading('timestamp', text='Data')
        self.tree.column('name', width=200)
        self.tree.column('class', width=150)
        self.tree.column('confidence', width=100)
        self.tree.column('timestamp', width=150)
        scrollbar = ttk.Scrollbar(f, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10,0), pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        return f

    def refresh_patterns_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for p in patterns_db.get_all_patterns_list():
            self.tree.insert('', tk.END, values=(p['name'], p['class'], f"{p['confidence']:.2f}", p['timestamp'][:19]))

    def _remove_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Uwaga", "Wybierz wzorzec.")
            return
        vals = self.tree.item(sel[0])['values']
        if messagebox.askyesno("Usuń", f"Usunąć '{vals[0]}' ({vals[1]})?"):
            patterns_db.remove(vals[0], vals[1])
            self.refresh_patterns_list()

    def _remove_all(self):
        if messagebox.askyesno("Usuń wszystkie", "Czy na pewno usunąć WSZYSTKIE wzorce?"):
            for cls in list(patterns_db.patterns.keys()):
                for p in patterns_db.patterns[cls][:]:
                    patterns_db.remove(p['name'], cls)
            self.refresh_patterns_list()
            messagebox.showinfo("OK", "Usunięto wszystkie wzorce.")

    # ---------- SZYBKA ANALIZA ----------
    def _create_quick_tab(self):
        f = ttk.Frame()
        ttk.Label(f, text="Analiza pojedynczego zdjęcia – unikalne ID wizualne.").pack(pady=10)
        ttk.Button(f, text="WYBIERZ ZDJĘCIE I ANALIZUJ", command=self._quick_analysis).pack(pady=20)
        self.quick_text = tk.Text(f, height=10, width=80)
        self.quick_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        return f

    def _quick_analysis(self):
        path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        analyzer = SingleImageInstanceAnalyzer(model_yolo, DEVICE, feature_extractor)
        try:
            annotated, detections = analyzer.process_image(path)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
            return
        out = f"analiza_{Path(path).stem}.jpg"
        cv2.imwrite(out, annotated)
        cv2.imshow("Szybka analiza", annotated)
        self.quick_text.delete(1.0, tk.END)
        classes = defaultdict(list)
        for d in detections:
            classes[d['class']].append(d['instance_id'])
        for cls in sorted(classes):
            ids = sorted(set(classes[cls]))
            self.quick_text.insert(tk.END, f"{cls}: {len(classes[cls])} obiektów (ID: {', '.join(map(str, ids))})\n")
        self.quick_text.insert(tk.END, f"\nZapisano: {out}\n")

# -------------------------- MAIN ----------------------------------------------
def main():
    print("System rozpoznawania obiektów – Jetson Orin Nano")
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()