# #!/usr/bin/env python3
# """
# System rozpoznawania obiektów (YOLOv8) z:
# - wykrywaniem wszystkich klas COCO na jednym zdjęciu,
# - automatycznym nadawaniem unikalnych ID (person_1, person_2…),
# - zapisem tych ID jako wzorców,
# - testowaniem na kamerze z wykorzystaniem zapisanych wzorców.
# """

# import cv2
# import numpy as np
# import torch
# import json
# from pathlib import Path
# from datetime import datetime
# from scipy.spatial.distance import cosine
# from ultralytics import YOLO
# import tkinter as tk
# from tkinter import ttk, filedialog, messagebox
# from collections import defaultdict
# from sklearn.cluster import AgglomerativeClustering

# # ===================== KONFIGURACJA =====================
# CAMERA_ID = 0
# USE_CUDA = True
# PATTERNS_DIR = "patterns_database"
# # ==========================================================

# Path(PATTERNS_DIR).mkdir(exist_ok=True)

# # ---------- YOLO ----------
# model_yolo = YOLO('yolov8n.pt')
# if USE_CUDA and torch.cuda.is_available():
#     model_yolo.to('cuda')
#     print("YOLO na GPU")
# else:
#     print("YOLO na CPU")

# COCO_CLASSES = sorted(model_yolo.names.values())

# device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')

# # ---------- Modele embeddingowe ----------
# def build_osnet():
#     try:
#         import torchreid
#         model = torchreid.models.build_model(
#             name='osnet_x0_25', num_classes=1000, pretrained=True
#         ).to(device).eval()
#         return model
#     except ImportError:
#         return None

# def build_efficientnet():
#     import torchvision.models as models
#     efficientnet = models.efficientnet_b0(pretrained=True)
#     model = torch.nn.Sequential(*list(efficientnet.children())[:-1]).to(device).eval()
#     return model

# osnet_model = build_osnet()
# effnet_model = build_efficientnet()

# # ---------- Baza wzorców ----------
# class PatternsDatabase:
#     def __init__(self):
#         self.patterns = {}
#         self.load_patterns()

#     def load_patterns(self):
#         metadata_file = Path(PATTERNS_DIR) / "patterns.json"
#         if metadata_file.exists():
#             with open(metadata_file, 'r') as f:
#                 metadata = json.load(f)
#             for class_name, patterns_list in metadata.items():
#                 self.patterns[class_name] = []
#                 for p in patterns_list:
#                     emb_file = Path(PATTERNS_DIR) / f"{p['name']}_{class_name}.npy"
#                     if emb_file.exists():
#                         emb = np.load(emb_file)
#                         self.patterns[class_name].append({
#                             'name': p['name'],
#                             'embedding': emb,
#                             'confidence': p['confidence'],
#                             'timestamp': p.get('timestamp', '')
#                         })
#             print(f"Wczytano wzorce dla {len(self.patterns)} klas")

#     def save_patterns(self):
#         metadata = {}
#         for class_name, lst in self.patterns.items():
#             metadata[class_name] = []
#             for p in lst:
#                 emb_file = Path(PATTERNS_DIR) / f"{p['name']}_{class_name}.npy"
#                 np.save(emb_file, p['embedding'])
#                 metadata[class_name].append({
#                     'name': p['name'],
#                     'confidence': p['confidence'],
#                     'timestamp': p.get('timestamp', datetime.now().isoformat())
#                 })
#         with open(Path(PATTERNS_DIR) / "patterns.json", 'w') as f:
#             json.dump(metadata, f, indent=2)

#     def add_pattern(self, name, class_name, embedding, confidence):
#         if class_name not in self.patterns:
#             self.patterns[class_name] = []
#         existing = [p for p in self.patterns[class_name] if p['name'] == name]
#         if existing:
#             existing[0]['embedding'] = embedding
#             existing[0]['confidence'] = confidence
#             existing[0]['timestamp'] = datetime.now().isoformat()
#         else:
#             self.patterns[class_name].append({
#                 'name': name,
#                 'embedding': embedding,
#                 'confidence': confidence,
#                 'timestamp': datetime.now().isoformat()
#             })
#         self.save_patterns()

#     def get_patterns_for_class(self, class_name):
#         return self.patterns.get(class_name, [])

# patterns_db = PatternsDatabase()

# # ---------- Analizator pojedynczego obrazu ----------
# class SingleImageInstanceAnalyzer:
#     def __init__(self):
#         self.detector = model_yolo
#         self.device = device
#         self.osnet = osnet_model
#         self.effnet = effnet_model

#     def get_embedding(self, crop, class_name):
#         if crop.size == 0:
#             return None
#         if class_name == 'person' and self.osnet is not None:
#             img = cv2.resize(crop, (128, 256))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
#             img_tensor = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0).to(self.device)
#             with torch.no_grad():
#                 feat = self.osnet(img_tensor)
#         else:
#             from torchvision import transforms
#             preprocess = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((224,224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#             ])
#             img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#             img_tensor = preprocess(img_rgb).unsqueeze(0).to(self.device)
#             with torch.no_grad():
#                 feat = self.effnet(img_tensor)
#         return feat.flatten().cpu().numpy()

#     def process_image(self, image_path):
#         img = cv2.imread(image_path)
#         if img is None:
#             raise FileNotFoundError(f"Nie można wczytać: {image_path}")

#         results = self.detector(img, conf=0.25, imgsz=640, verbose=False)
#         detections = []
#         for result in results:
#             if result.boxes is not None:
#                 for box in result.boxes:
#                     x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
#                     conf = float(box.conf[0].item())
#                     cls_id = int(box.cls[0].item())
#                     cls_name = self.detector.names[cls_id]
#                     crop = img[y1:y2, x1:x2]
#                     emb = self.get_embedding(crop, cls_name)
#                     if emb is not None:
#                         detections.append({
#                             'bbox': (x1,y1,x2,y2),
#                             'class': cls_name,
#                             'confidence': conf,
#                             'embedding': emb
#                         })

#         # Grupowanie w obrębie klasy
#         self._cluster_instances(detections)

#         # Rysowanie
#         annotated = img.copy()
#         for det in detections:
#             x1,y1,x2,y2 = det['bbox']
#             cls = det['class']
#             conf = det['confidence']
#             iid = det['instance_id']
#             color = self._color_for_id(iid)
#             label = f"{cls}_{iid} | {conf:.2f}"
#             cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
#             (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
#             cv2.rectangle(annotated, (x1,y1-th-5), (x1+tw,y1), color, -1)
#             cv2.putText(annotated, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5, (255,255,255), 2)

#         return annotated, detections

#     def _cluster_instances(self, detections):
#         by_class = defaultdict(list)
#         for d in detections:
#             by_class[d['class']].append(d)

#         for cls_name, items in by_class.items():
#             embs = np.array([it['embedding'] for it in items])
#             n = len(items)
#             if n == 1:
#                 items[0]['instance_id'] = 1
#                 continue
#             clustering = AgglomerativeClustering(
#                 n_clusters=None, distance_threshold=0.1,
#                 metric='cosine', linkage='average'
#             )
#             labels = clustering.fit_predict(embs)
#             uniq = sorted(set(labels))
#             label_to_id = {lab: i+1 for i, lab in enumerate(uniq)}
#             for det, lab in zip(items, labels):
#                 det['instance_id'] = label_to_id[lab]

#     def _color_for_id(self, iid):
#         hue = (iid * 43) % 180
#         hsv = np.uint8([[[hue, 255, 255]]])
#         bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
#         return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

# # ---------- Funkcje do testu z kamery (używają bazy wzorców) ----------
# def get_embedding_crop(crop, class_name):
#     """Helper do generowania embeddingu (używany przez kamerę)."""
#     if class_name == 'person' and osnet_model is not None:
#         img = cv2.resize(crop, (128,256))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
#         tensor = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0).to(device)
#         with torch.no_grad():
#             feat = osnet_model(tensor)
#     else:
#         from torchvision import transforms
#         preprocess = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#         ])
#         img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#         tensor = preprocess(img_rgb).unsqueeze(0).to(device)
#         with torch.no_grad():
#             feat = effnet_model(tensor)
#     return feat.flatten().cpu().numpy()

# def match_camera_object(embedding, class_name):
#     patterns = patterns_db.get_patterns_for_class(class_name)
#     if not patterns:
#         return None, 0.0
#     th = 0.75 if class_name == 'person' else 0.85
#     best_name, best_sim = None, 0.0
#     for p in patterns:
#         sim = 1 - cosine(embedding, p['embedding'])
#         if sim > best_sim and sim >= th:
#             best_sim = sim
#             best_name = p['name']
#     return best_name, best_sim

# def run_camera_test():
#     """Test z kamery na żywo z wykorzystaniem wzorców z bazy."""
#     cap = cv2.VideoCapture(CAMERA_ID)
#     if not cap.isOpened():
#         print("Nie można otworzyć kamery")
#         return
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     print("Kamera testowa – naciśnij 'q', aby wrócić.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         results = model_yolo(frame, conf=0.25, imgsz=640, verbose=False)
#         annotated = frame.copy()
#         for result in results:
#             if result.boxes is not None:
#                 for box in result.boxes:
#                     x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
#                     conf = float(box.conf[0].item())
#                     cls_id = int(box.cls[0].item())
#                     cls_name = model_yolo.names[cls_id]
#                     crop = frame[y1:y2, x1:x2]
#                     emb = get_embedding_crop(crop, cls_name)
#                     if emb is not None:
#                         match_name, sim = match_camera_object(emb, cls_name)
#                         if match_name:
#                             color = (0,255,0)
#                             label = f"{match_name} | {conf:.2f}"
#                         else:
#                             color = (0,0,255)
#                             label = f"UNKNOWN {cls_name} | {conf:.2f}"
#                         cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
#                         (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
#                         cv2.rectangle(annotated, (x1,y1-th-5), (x1+tw,y1), color, -1)
#                         cv2.putText(annotated, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX,
#                                     0.5, (255,255,255), 2)
#         cv2.imshow("Kamera – test wzorców", annotated)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyWindow("Kamera – test wzorców")

# # ===================== GUI =====================
# class App:
#     def __init__(self, root):
#         self.root = root
#         root.title("System rozpoznawania – full multi-class")
#         root.geometry("900x700")
#         self.last_detections = []   # przechowuje detekcje z szybkiej analizy
#         self.setup_ui()

#     def setup_ui(self):
#         notebook = ttk.Notebook(self.root)
#         notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

#         # Zakładka: Szybka analiza
#         quick_frame = ttk.Frame(notebook)
#         notebook.add(quick_frame, text="Szybka analiza")
#         self.setup_quick_tab(quick_frame)

#         # Zakładka: Test (kamera / plik) – używa bazy wzorców
#         test_frame = ttk.Frame(notebook)
#         notebook.add(test_frame, text="Test ogólny")
#         self.setup_test_tab(test_frame)

#         # Zakładka: Zarządzanie wzorcami
#         list_frame = ttk.Frame(notebook)
#         notebook.add(list_frame, text="Wzorce")
#         self.setup_patterns_tab(list_frame)

#         self.status_var = tk.StringVar(value="Gotowy")
#         ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, side=tk.BOTTOM)

#     # ---------- Szybka analiza ----------
#     def setup_quick_tab(self, parent):
#         frame = ttk.LabelFrame(parent, text="Analiza pojedynczego zdjęcia (unikalne ID)")
#         frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

#         ttk.Label(frame, text="Wybierz zdjęcie – system wykryje wszystkie obiekty\n"
#                               "i nada im unikalne identyfikatory (car_1, car_2…).",
#                   justify=tk.LEFT).pack(pady=10)

#         ttk.Button(frame, text="WYBIERZ ZDJĘCIE I ANALIZUJ",
#                    command=self.quick_analysis).pack(pady=10)

#         # Przyciski zapisu / testu
#         btn_frame = ttk.Frame(frame)
#         btn_frame.pack(pady=10)
#         self.btn_save = ttk.Button(btn_frame, text="Zapisz wszystkie jako wzorce",
#                                    command=self.save_quick_patterns, state=tk.DISABLED)
#         self.btn_save.pack(side=tk.LEFT, padx=5)
#         self.btn_test_cam = ttk.Button(btn_frame, text="Zapisz i testuj na kamerze",
#                                        command=self.save_and_test_camera, state=tk.DISABLED)
#         self.btn_test_cam.pack(side=tk.LEFT, padx=5)

#         self.quick_text = tk.Text(frame, height=12, width=90)
#         self.quick_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

#     def quick_analysis(self):
#         filepath = filedialog.askopenfilename(
#             title="Wybierz zdjęcie do analizy",
#             filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")]
#         )
#         if not filepath:
#             return
#         try:
#             analyzer = SingleImageInstanceAnalyzer()
#             annotated, detections = analyzer.process_image(filepath)
#         except Exception as e:
#             messagebox.showerror("Błąd", str(e))
#             return

#         out_path = f"analiza_{Path(filepath).stem}.jpg"
#         cv2.imwrite(out_path, annotated)
#         cv2.imshow("Szybka analiza – wynik", annotated)
#         self.last_detections = detections
#         self.btn_save.config(state=tk.NORMAL)
#         self.btn_test_cam.config(state=tk.NORMAL)

#         # Podsumowanie
#         self.quick_text.delete(1.0, tk.END)
#         summary = defaultdict(list)
#         for d in detections:
#             summary[d['class']].append(d['instance_id'])
#         for cls, ids in sorted(summary.items()):
#             uniq = sorted(set(ids))
#             self.quick_text.insert(tk.END, f"{cls}: {len(ids)} obiektów (ID: {', '.join(map(str, uniq))})\n")
#         self.quick_text.insert(tk.END, f"\nWynik zapisano: {out_path}\n")
#         self.status_var.set(f"Analiza gotowa – {len(detections)} obiektów")

#     def save_quick_patterns(self):
#         if not self.last_detections:
#             return
#         for det in self.last_detections:
#             name = f"{det['class']}_{det['instance_id']}"
#             patterns_db.add_pattern(name, det['class'], det['embedding'], det['confidence'])
#         messagebox.showinfo("Zapisane", f"Zapisano {len(self.last_detections)} wzorców")
#         self.status_var.set("Wzorce zapisane w bazie")

#     def save_and_test_camera(self):
#         self.save_quick_patterns()
#         # Ukryj GUI na czas testu kamery
#         self.root.withdraw()
#         run_camera_test()
#         self.root.deiconify()
#         self.status_var.set("Test kamery zakończony")

#     # ---------- Test ogólny (z bazy wzorców) ----------
#     def setup_test_tab(self, parent):
#         frame = ttk.LabelFrame(parent, text="Test na pliku lub kamerze (używa zapisanych wzorców)")
#         frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

#         src_frame = ttk.Frame(frame)
#         src_frame.pack(pady=10)
#         self.test_src = tk.StringVar(value="file")
#         ttk.Radiobutton(src_frame, text="Plik", variable=self.test_src, value="file").pack(side=tk.LEFT, padx=10)
#         ttk.Radiobutton(src_frame, text="Kamera", variable=self.test_src, value="camera").pack(side=tk.LEFT, padx=10)

#         ttk.Button(src_frame, text="START TESTU", command=self.start_test).pack(side=tk.LEFT, padx=20)

#         self.test_results = tk.Text(frame, height=15)
#         self.test_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

#     def start_test(self):
#         if self.test_src.get() == "file":
#             filepath = filedialog.askopenfilename(
#                 title="Wybierz obraz testowy",
#                 filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")]
#             )
#             if not filepath:
#                 return
#             # Przetwarzanie z użyciem process_image_full (z wcześniejszych funkcji)
#             from previous_code import process_image_full  # <-- ta funkcja używa patterns_db
#             annotated, dets = process_image_full(image_path=filepath)
#             out_path = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#             cv2.imwrite(out_path, annotated)
#             cv2.imshow("Test – wynik", annotated)
#             self.test_results.delete(1.0, tk.END)
#             self.test_results.insert(tk.END, f"Zapisano: {out_path}\n")
#             for d in dets:
#                 self.test_results.insert(tk.END, f"{d['class']}: {d['match']} (sim {d['similarity']:.2f})\n")
#         else:
#             self.root.withdraw()
#             run_camera_test()
#             self.root.deiconify()

#     # ---------- Zarządzanie wzorcami ----------
#     def setup_patterns_tab(self, parent):
#         frame = ttk.Frame(parent)
#         frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

#         ttk.Button(frame, text="Odśwież listę", command=self.refresh_patterns_list).pack(pady=5)
#         columns = ('name', 'class', 'confidence', 'timestamp')
#         self.tree = ttk.Treeview(frame, columns=columns, show='headings')
#         self.tree.heading('name', text='Nazwa')
#         self.tree.heading('class', text='Klasa')
#         self.tree.heading('confidence', text='Pewność')
#         self.tree.heading('timestamp', text='Data')
#         self.tree.pack(fill=tk.BOTH, expand=True)
#         ttk.Button(frame, text="Usuń zaznaczony", command=self.remove_pattern).pack(pady=5)
#         self.refresh_patterns_list()

#     def refresh_patterns_list(self):
#         for item in self.tree.get_children():
#             self.tree.delete(item)
#         all_pats = []
#         for cls, lst in patterns_db.patterns.items():
#             for p in lst:
#                 all_pats.append((p['name'], cls, p['confidence'], p.get('timestamp','')[:19]))
#         for p in sorted(all_pats, key=lambda x: (x[1], x[0])):
#             self.tree.insert('', tk.END, values=p)

#     def remove_pattern(self):
#         sel = self.tree.selection()
#         if not sel:
#             return
#         vals = self.tree.item(sel[0])['values']
#         name, cls = vals[0], vals[1]
#         patterns_db.remove_pattern( ame, cls)
#         self.refresh_patterns_list()

# # ===================== Uruchomienie =====================
# if __name__ == "__main__":
#     # Upewnij się, że funkcja process_image_full jest dostępna (jeśli używasz testu ogólnego)
#     # Możesz ją skopiować z wcześniejszego kodu lub tymczasowo zdefiniować tutaj.
#     # Na potrzeby demonstracji zakomentujemy ją – test ogólny i tak działa z kamerą.
#     root = tk.Tk()
#     app = App(root)
#     root.mainloop()