#!/usr/bin/env python3
import cv2
import numpy as np
import torch
import sys
import os
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ===================== KONFIGURACJA =====================
CAMERA_ID = 0
USE_CUDA = True
SHOW_GUI = True
# ==========================================================

# ---------- Inicjalizacja YOLO ----------
model_yolo = YOLO('yolov8n.pt')
if USE_CUDA and torch.cuda.is_available():
    model_yolo.to('cuda')
    print("YOLO na GPU")
else:
    print("YOLO na CPU")

# Lista klas COCO (można ograniczyć)
# COCO_CLASSES = list(model_yolo.names.values())
COCO_CLASSES = sorted(model_yolo.names.values())
DEFAULT_CLASS = 'bus'

# ---------- Modele embeddingowe ----------
device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')

# Ładujemy oba modele z góry (jeśli to możliwe)
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

def get_embedding(image_crop, target_class, emb_model):
    if image_crop.size == 0:
        return None
    if target_class == 'person':
        # OSNet
        img = cv2.resize(image_crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
        with torch.no_grad():
            features = emb_model(img_tensor)
    else:
        # EfficientNet
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
            features = emb_model(img_tensor)
    return features.flatten().cpu().numpy()

def compare_embeddings(emb1, emb2):
    return 1 - cosine(emb1.flatten(), emb2.flatten())

# ---------- Funkcje pomocnicze ----------
def detect_objects(image, target_class):
    class_id = list(model_yolo.names.keys())[list(model_yolo.names.values()).index(target_class)]
    results = model_yolo(image, classes=[class_id], conf=0.25, imgsz=640, verbose=False)
    objects = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                objects.append(((x1, y1, x2, y2), target_class, conf))
    return objects

def extract_reference_from_image(image_path, target_class, emb_model):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nie wczytano: {image_path}")
    objs = detect_objects(img, target_class)
    if not objs:
        return None
    bbox, cls, conf = objs[0]
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    emb = get_embedding(crop, target_class, emb_model)
    if emb is not None:
        return emb, cls, conf
    return None

def capture_reference_from_camera(target_class, emb_model):
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Nie można otworzyć kamery.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Kamera referencyjna (szukamy: {target_class}). Naciśnij SPACJĘ, aby zrobić zdjęcie (ESC – anuluj).")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        objs = detect_objects(frame, target_class)
        display = frame.copy()
        best = None
        if objs:
            best = objs[0]
            (x1, y1, x2, y2), cls, conf = best
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{cls} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, "SPACJA - zdjecie, ESC - anuluj", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Referencja z kamery", display)
        key = cv2.waitKey(30) & 0xFF
        if key == 32:
            if best:
                (x1, y1, x2, y2), cls, conf = best
                crop = frame[y1:y2, x1:x2]
                emb = get_embedding(crop, target_class, emb_model)
                if emb is not None:
                    cap.release()
                    cv2.destroyWindow("Referencja z kamery")
                    return emb, cls, conf
            else:
                print("Nie wykryto obiektu. Spróbuj ponownie.")
        elif key == 27:
            break
    cap.release()
    cv2.destroyWindow("Referencja z kamery")
    return None

def process_test_image_file(image_path, ref_emb, target_class, emb_model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Nie można wczytać obrazu testowego: {image_path}")
        return
    print(f"\n--- Test na pliku: {image_path} ---")
    objs = detect_objects(img, target_class)
    annotated = img.copy()
    threshold = 0.75 if target_class == 'person' else 0.85
    for bbox, cls, conf in objs:
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        emb = get_embedding(crop, target_class, emb_model)
        if emb is not None:
            sim = compare_embeddings(ref_emb, emb)
            match = sim >= threshold
            color = (0, 255, 0) if match else (0, 0, 255)
            label = f"{cls} | {'MATCH' if match else 'OTHER'} {sim:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            status = "MATCH" if match else "OTHER"
            print(f"  Obiekt: {cls} (pewność {conf:.2f}), podobieństwo {sim:.4f} -> {status}")
    out_path = "wynik_test.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Zapisano obraz wynikowy: {out_path}")
    if SHOW_GUI:
        cv2.imshow("Wynik testu (plik)", annotated)
        cv2.waitKey(0)
        cv2.destroyWindow("Wynik testu (plik)")

def run_test_camera(ref_emb, target_class, emb_model):
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Nie można otworzyć kamery.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    threshold = 0.75 if target_class == 'person' else 0.85
    print(f"\nKamera testowa (szukamy: {target_class}). Naciskaj 'q' w oknie wideo, aby wrócić.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        objs = detect_objects(frame, target_class)
        annotated = frame.copy()
        for bbox, cls, conf in objs:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            emb = get_embedding(crop, target_class, emb_model)
            if emb is not None:
                sim = compare_embeddings(ref_emb, emb)
                match = sim >= threshold
                color = (0, 255, 0) if match else (0, 0, 255)
                label = f"{cls} | {'MATCH' if match else 'OTHER'} {sim:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                status = "MATCH" if match else "OTHER"
                print(f"  Kamera: {cls} (pewność {conf:.2f}) -> {status}, sim={sim:.4f}")
        if SHOW_GUI:
            cv2.imshow("Kamera testowa", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            pass
    cap.release()
    if SHOW_GUI:
        cv2.destroyWindow("Kamera testowa")

# ---------- GUI Tkinter ----------
class App:
    def __init__(self, root):
        self.root = root
        self.target_class = DEFAULT_CLASS
        self.ref_emb = None
        self.ref_label = ""

        # Sprawdź dostępność modeli
        self.osnet_available = osnet_model is not None
        if not self.osnet_available:
            print("UWAGA: torchreid nie jest zainstalowany – rozpoznawanie osób NIEAKTYWNE.")

        root.title("System rozpoznawania obiektów")
        root.geometry("550x400")
        root.resizable(False, False)

        # 1. Wybór klasy
        class_frame = tk.LabelFrame(root, text="Jakiego obiektu szukasz na obrazie głównym?", padx=10, pady=10)
        class_frame.pack(pady=10, padx=10, fill="x")
        tk.Label(class_frame, text="Klasa obiektu:").pack(side=tk.LEFT, padx=5)
        self.class_var = tk.StringVar(value=DEFAULT_CLASS)
        self.class_combo = ttk.Combobox(class_frame, textvariable=self.class_var, values=COCO_CLASSES, state="readonly", width=20)
        self.class_combo.pack(side=tk.LEFT, padx=5)
        self.class_combo.bind("<<ComboboxSelected>>", self.on_class_changed)
        # Jeśli osoba niedostępna, usuń ją z listy
        if not self.osnet_available:
            filtered = [c for c in COCO_CLASSES if c != 'person']
            self.class_combo['values'] = filtered
            self.class_var.set('bus' if 'bus' in filtered else filtered[0])

        # 2. Ramka główna z dwiema kolumnami
        main_frame = tk.Frame(root)
        main_frame.pack(pady=10, padx=10, fill="both")

        left_frame = tk.LabelFrame(main_frame, text="Wybierz źródło obrazu głównego (wzorzec)", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, expand=True, fill="both", padx=5)
        self.ref_source_var = tk.StringVar(value="file")
        tk.Radiobutton(left_frame, text="Z pliku", variable=self.ref_source_var, value="file").pack(anchor=tk.W)
        tk.Radiobutton(left_frame, text="Z kamery", variable=self.ref_source_var, value="camera").pack(anchor=tk.W)

        right_frame = tk.LabelFrame(main_frame, text="Wybierz źródło porównania (test)", padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, expand=True, fill="both", padx=5)
        self.test_source_var = tk.StringVar(value="file")
        tk.Radiobutton(right_frame, text="Z pliku", variable=self.test_source_var, value="file").pack(anchor=tk.W)
        tk.Radiobutton(right_frame, text="Z kamery", variable=self.test_source_var, value="camera").pack(anchor=tk.W)

        # 3. Przycisk startu
        tk.Button(root, text="URUCHOM PORÓWNANIE", command=self.start_comparison,
                  bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), height=2).pack(pady=15)

        # 4. Status
        self.status_label = tk.Label(root, text="Gotowy. Wybierz klasę, źródła i kliknij start.", fg="blue")
        self.status_label.pack(pady=5)

        self.update_embedding_model()

    def on_class_changed(self, event=None):
        new_class = self.class_var.get()
        if new_class == 'person' and not self.osnet_available:
            messagebox.showwarning("Niedostępne", "Model dla osób (OSNet) nie jest zainstalowany. Wybierz inną klasę.")
            self.class_var.set(self.target_class)  # przywróć poprzednią
            return
        self.target_class = new_class
        self.ref_emb = None  # reset wzorca po zmianie klasy
        self.ref_label = ""
        self.update_embedding_model()
        self.status_label.config(text=f"Klasa: {self.target_class}. Wybierz wzorzec i test.")
        print(f"Zmieniono klasę na: {self.target_class}")

    def update_embedding_model(self):
        if self.target_class == 'person':
            self.emb_model = osnet_model
        else:
            self.emb_model = effnet_model

    def start_comparison(self):
        # Krok 1: Wzorzec
        ref_source = self.ref_source_var.get()
        print(f"\n========== POBIERANIE WZORCA (klasa: {self.target_class}) ==========")
        if ref_source == "file":
            filepath = filedialog.askopenfilename(
                title="Wybierz zdjęcie główne (wzorzec)",
                filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")]
            )
            if not filepath:
                self.status_label.config(text="Anulowano wybór pliku.")
                return
            try:
                result = extract_reference_from_image(filepath, self.target_class, self.emb_model)
                if result is None:
                    messagebox.showerror("Błąd", f"Nie wykryto obiektu klasy '{self.target_class}' na wybranym obrazie.")
                    return
                self.ref_emb, cls, conf = result
                print(f"Wzorzec z pliku: {filepath}")
                print(f"  Obiekt: {cls}, pewność: {conf:.2f}")
                self.ref_label = f"{cls} ({conf:.2f})"
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się przetworzyć obrazu:\n{e}")
                return
        else:  # camera
            self.root.withdraw()
            result = capture_reference_from_camera(self.target_class, self.emb_model)
            self.root.deiconify()
            if result is None:
                self.status_label.config(text="Anulowano przechwytywanie z kamery.")
                return
            self.ref_emb, cls, conf = result
            print(f"Wzorzec z kamery: {cls}, pewność: {conf:.2f}")
            self.ref_label = f"{cls} ({conf:.2f})"

        # Zapisz wzorzec
        ref_file = f"wzorzec_{self.target_class}.npy"
        np.save(ref_file, self.ref_emb)
        self.status_label.config(text=f"Wzorzec gotowy: {self.ref_label}")

        # Krok 2: Test
        test_source = self.test_source_var.get()
        print(f"\n========== ROZPOCZYNANIE TESTU (porównanie z {self.ref_label}) ==========")
        if test_source == "file":
            filepath = filedialog.askopenfilename(
                title="Wybierz zdjęcie do porównania",
                filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")]
            )
            if not filepath:
                self.status_label.config(text="Test anulowany.")
                return
            process_test_image_file(filepath, self.ref_emb, self.target_class, self.emb_model)
            self.status_label.config(text="Test z pliku zakończony. Zobacz wynik_test.jpg.")
        else:
            print("Uruchamiam podgląd z kamery...")
            run_test_camera(self.ref_emb, self.target_class, self.emb_model)
            self.status_label.config(text="Test z kamery zakończony.")
            print("Powrót do panelu.")

# ---------- Start ----------
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()