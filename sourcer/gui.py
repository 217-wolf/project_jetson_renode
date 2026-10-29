#!/usr/bin/env python3
"""
Interfejs graficzny systemu - zakładki do zarządzania wzorcami, testów i szybkiej analizy.
Wykorzystuje moduły funkcyjne.
"""
#import pythona
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Importy modułów systemu
from detector import ObjectDetector
from extractor import FeatureExtractor
from database import PatternsDatabase
from matcher import ObjectMatcher
from visualizer import Visualizer
from camera import CameraManager
from analyzer import ImageAnalyzer
from reid_tracker import PersistentReIDTracker

#import zewnętrzny
import cv2

logger = logging.getLogger(__name__) #informacja o nazwie modułu

class MainApplication:
    """Główna klasa aplikacji GUI."""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("System Identyfikacji Obiektów- SIO")
        self.root.geometry("900x650")

        # Komponenty systemu (leniwa inicjalizacja)
        self.detector = None
        self.extractor = None
        self.database = PatternsDatabase()
        self.matcher = None
        self.visualizer = Visualizer()
        self.camera = CameraManager()
        self.analyzer = None

        self._init_components()
        self._setup_ui()
        self.refresh_patterns_list()

    def _init_components(self):
        """Inicjalizacja komponentów, które mogą być czasochłonne."""
        self.detector = ObjectDetector()
        self.extractor = FeatureExtractor()
        self.matcher = ObjectMatcher(self.database)
        self.analyzer = ImageAnalyzer(self.detector, self.extractor)

    def _setup_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Zakładki
        nb.add(self._create_add_tab(), text="Dodaj wzorce")
        nb.add(self._create_test_tab(), text="Test działania")
        nb.add(self._create_list_tab(), text="Lista wzorców")
        nb.add(self._create_quick_tab(), text="Szybka analiza")

        # Pasek statusu
        self.status_var = tk.StringVar(value="Gotowy.")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)\
            .pack(fill=tk.X, side=tk.BOTTOM)

    # Zakładka - dodawania wzorców --------------------------------------------------
    def _create_add_tab(self):
        frame = ttk.Frame()

        # Pojedynczy wzorzec
        sf = ttk.LabelFrame(frame, text="Dodaj pojedynczy wzorzec")
        sf.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(sf, text="Nazwa:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.name_var = tk.StringVar()
        ttk.Entry(sf, textvariable=self.name_var, width=30).grid(row=0, column=1, padx=5)

        ttk.Label(sf, text="Klasa:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.class_var = tk.StringVar()
        self.class_combo = ttk.Combobox(sf, textvariable=self.class_var,
                                        values=self.detector.class_names if self.detector else [],
                                        state="readonly", width=27)
        self.class_combo.grid(row=1, column=1, padx=5)

        self.src_var = tk.StringVar(value="file")
        ttk.Radiobutton(sf, text="Plik", variable=self.src_var, value="file")\
            .grid(row=2, column=1, sticky=tk.W)
        ttk.Radiobutton(sf, text="Kamera", variable=self.src_var, value="camera")\
            .grid(row=3, column=1, sticky=tk.W)

        ttk.Button(sf, text="DODAJ WZORZEC", command=self._add_single_pattern)\
            .grid(row=4, column=0, columnspan=2, pady=10)

        # Wiele wzorców z obrazu
        mf = ttk.LabelFrame(frame, text="Dodaj wszystkie obiekty z obrazu jako wzorce")
        mf.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(mf, text="Każdy wykryty obiekt zostanie zapisany jako klasa_001, klasa_002, ... etc.")\
            .pack(pady=5)
        ttk.Button(mf, text="Z PLIKU", command=self._add_all_file).pack(pady=3)
        ttk.Button(mf, text="Z KAMERY", command=self._add_all_camera).pack(pady=3)

        return frame

    def _add_single_pattern(self):
        name = self.name_var.get().strip()
        cls = self.class_var.get().strip()
        if not name or not cls:
            messagebox.showerror("Błąd", "Podaj nazwę i klasę")
            return

        if cls not in self.detector.class_names:
            messagebox.showerror("Błąd", "Nieznana klasa")
            return

        if self.src_var.get() == "file":
            path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.jpeg *.png")])
            if not path: return
            img = cv2.imread(path)
        else:
            self.root.withdraw()
            img = self.camera.capture_photo("Dodaj wzorzec")
            self.root.deiconify()

        if img is None:
            return

        # Detekcja tylko wybranej klasy
        class_id = self.detector.name_to_id[cls]
        detections = self.detector.detect(img, classes=[cls])
        if not detections:
            messagebox.showerror("Błąd", f"Nie wykryto obiektu klasy '{cls}'")
            return

        # Bierzemy pierwszy obiekt
        obj = detections[0]
        x1,y1,x2,y2 = obj['bbox']
        crop = img[y1:y2, x1:x2]
        emb = self.extractor.extract(crop, cls)

        if emb is not None:
            self.database.add_pattern(name, cls, emb, obj['confidence'])
            messagebox.showinfo("Sukces", f"Wzorzec '{name}' dodany")
            self.refresh_patterns_list()
        else:
            messagebox.showerror("Błąd", "Nie udało się wyekstrahować cech")

    def _add_all_file(self):
        path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.jpeg *.png")])
        if not path: return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Błąd", "Nie można wczytać obrazu")
            return
        self._add_all_from_image(img)

    def _add_all_camera(self):
        self.root.withdraw()
        img = self.camera.capture_photo("Dodaj wszystkie obiekty")
        self.root.deiconify()
        if img is not None:
            self._add_all_from_image(img)

    def _add_all_from_image(self, img):
        detections = self.detector.detect(img)
        added = 0
        for i, det in enumerate(detections):
            x1,y1,x2,y2 = det['bbox']
            crop = img[y1:y2, x1:x2]
            emb = self.extractor.extract(crop, det['class'])

            if emb is not None:
                name = f"{det['class']}_{i+1:03d}"
                self.database.add_pattern(name, det['class'], emb, det['confidence'])
                added += 1
        messagebox.showinfo("Gotowe", f"Dodano {added} wzorców")
        self.refresh_patterns_list()

    # Zakładka - Testu -------------------------------------------------------------------------
    def _create_test_tab(self):
        f = ttk.Frame()
        sf = ttk.LabelFrame(f, text="Źródło testu")
        sf.pack(fill=tk.X, padx=10, pady=5)
        self.test_src = tk.StringVar(value="image")
        ttk.Radiobutton(sf, text="Obraz", variable=self.test_src, value="image")\
            .pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(sf, text="Plik wideo", variable=self.test_src, value="video")\
            .pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(sf, text="Kamera", variable=self.test_src, value="camera")\
            .pack(side=tk.LEFT, padx=10)
        ttk.Button(sf, text="ROZPOCZNIJ TEST", command=self._start_test)\
            .pack(side=tk.LEFT, padx=20)

        rf = ttk.LabelFrame(f, text="Wyniki")
        rf.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.results_text = tk.Text(rf, height=12, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        return f

    def _start_test(self):
        self.results_text.delete(1.0, tk.END)
        src = self.test_src.get()
        if src == "image":
            path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.jpeg *.png")])
            if not path:
                return
            img = cv2.imread(path)
            if img is not None:
                self._process_test_image(img, path)
        elif src == "video":
            path = filedialog.askopenfilename(
                title="Wybierz plik wideo",
                filetypes=[("Pliki wideo", "*.mp4 *.avi *.mkv *.mov"), ("Wszystkie", "*.*")]
            )
            if not path:
                return
            self.root.withdraw()
            self._run_live_test(source=path)
            self.root.deiconify()
        else:  # camera
            self.root.withdraw()
            self._run_live_test(source=None)
            self.root.deiconify()

    def _process_test_image(self, img, path=None):
        detections = self.detector.detect(img)
        for detection in detections:
            x1,y1,x2,y2 = detection['bbox']
            crop = img[y1:y2, x1:x2]
            embedding = self.extractor.extract(crop, detection['class'])
            if embedding is not None:
                match, sim = self.matcher.match(embedding, detection['class'])
                detection['match'] = match or 'UNKNOWN'
                detection['similarity'] = sim
            else:
                detection['match'] = 'UNKNOWN'
                detection['similarity'] = 0.0

        annotated = self.visualizer.draw_detections(img, detections)
        out = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(out, annotated)
        self._display_results(detections, out)
        cv2.imshow("Wynik testu", annotated)
        cv2.waitKey(0)
        cv2.destroyWindow("Wynik testu")

    def _run_live_test(self, source=None):
        """Ujednolicony test live – source=None oznacza kamerę, source=ścieżka oznacza plik wideo."""
        if source is not None:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                messagebox.showerror("Błąd", f"Nie można otworzyć pliku wideo:\n{source}")
                return
            read_func = cap.read
            close_func = cap.release
            window_title = f"ReID: {source}"
            logger.info(f"Otworzono plik wideo: {source}")
        else:
            if not self.camera.open():
                messagebox.showerror("Błąd", "Nie można otworzyć kamery")
                return
            read_func = self.camera.read_frame
            close_func = self.camera.close
            window_title = "Test z kamery"
            logger.info("Kamera live - naciśnij 'Q' / 'Esc' aby zakończyć")

        reid_tracker = PersistentReIDTracker()
        while True:
            ret, frame = read_func()
            if not ret:
                break
            detections = self.detector.detect(frame)
            trackable = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                crop = frame[y1:y2, x1:x2]
                emb = self.extractor.extract(crop, det['class'])
                if emb is not None:
                    det['embedding'] = emb
                    trackable.append(det)
                    match, sim = self.matcher.match(emb, det['class'])
                    det['match'] = match or 'NIEZNANY'
                    det['similarity'] = sim
                else:
                    det['match'] = 'NIEZNANY'
                    det['similarity'] = 0.0

            reid_tracker.update(trackable)
            annotated = self.visualizer.draw_detections(frame, detections)

            cv2.imshow(window_title, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'Q' lub ESC
                break

        close_func()
        cv2.destroyWindow(window_title)

    def _display_results(self, detections, out_path):
        self.results_text.insert(tk.END, f"Zapisano: {out_path}\n{'='*60}\n")
        by_class = defaultdict(list)
        for d in detections:
            by_class[d['class']].append(d)
        for cls in sorted(by_class):
            self.results_text.insert(tk.END, f"\n--- {cls.upper()} ---\n")
            for d in by_class[cls]:
                status = "✓" if d['match'] != 'UNKNOWN' else "✗"
                line = f"{status} {d['match']} (pewność: {d['confidence']:.2f}"
                if d['similarity'] > 0:
                    line += f", podobieństwo: {d['similarity']:.4f}"
                line += ")\n"
                self.results_text.insert(tk.END, line)
        total = len(detections)
        matched = sum(1 for d in detections if d['match'] != 'UNKNOWN')
        self.results_text.insert(tk.END, f"\nRAZEM: {total}, rozpoznanych: {matched}\n")
        self.status_var.set(f"Test zakończony – {matched}/{total}")

    #Zakładka listy wzorców ----------------------------------------------------------------------------------
    def _create_list_tab(self):
        f = ttk.Frame()
        btnf = ttk.Frame(f)
        btnf.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btnf, text="Odśwież", command=self.refresh_patterns_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(btnf, text="Usuń zaznaczony", command=self._remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btnf, text="Usuń wszystkie", command=self._remove_all).pack(side=tk.LEFT, padx=5)

        cols = ('name','class','confidence','timestamp')
        self.tree = ttk.Treeview(f, columns=cols, show='headings')
        self.tree.heading('name', text='Nazwa')
        self.tree.heading('class', text='Klasa')
        self.tree.heading('confidence', text='Pewność')
        self.tree.heading('timestamp', text='Data dodania')
        self.tree.column('name', width=200)
        self.tree.column('class', width=150)
        self.tree.column('confidence', width=100)
        self.tree.column('timestamp', width=150)

        scrollbar = ttk.Scrollbar(f, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10,0), pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        return f

    def refresh_patterns_list(self):
        if hasattr(self, 'tree'):
            for item in self.tree.get_children():
                self.tree.delete(item)
            for p in self.database.get_all_patterns():
                self.tree.insert('', tk.END, values=(p['name'], p['class'],
                                  f"{p['confidence']:.2f}", p['timestamp'][:19]))

    def _remove_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Uwaga", "Wybierz wzorzec")
            return
        vals = self.tree.item(sel[0])['values']
        if messagebox.askyesno("Usuń", f"Usunąć '{vals[0]}' ({vals[1]})?"):
            self.database.remove_pattern(vals[0], vals[1])
            self.refresh_patterns_list()

    def _remove_all(self):
        if messagebox.askyesno("Usuń wszystkie", "Czy na pewno usunąć WSZYSTKIE wzorce?"):
            self.database.clear()
            self.refresh_patterns_list()
            messagebox.showinfo("Gotowe", "Wszystkie wzorce usunięte")

    #Zakładka: Szybka analiza ------------------------
    def _create_quick_tab(self):
        f = ttk.Frame()
        ttk.Label(f, text="Analiza pojedynczego zdjęcia – unikalne ID wizualne.")\
            .pack(pady=10)
        ttk.Button(f, text="Wybierz zdjęcie i analizuj", command=self._quick_analysis)\
            .pack(pady=15)
        self.quick_text = tk.Text(f, height=10, width=80)
        self.quick_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        return f

    def _quick_analysis(self):
        path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.jpeg *.png")])
        if not path: return
        try:
            img, detections = self.analyzer.analyze(path)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
            return

        annotated = self.visualizer.draw_instance_ids(img, detections)
        out = f"analiza_{Path(path).stem}.jpg"
        cv2.imwrite(out, annotated)
        cv2.imshow("Szybka analiza – ID instancji", annotated)

        self.quick_text.delete(1.0, tk.END)
        summary = defaultdict(list)
        for d in detections:
            summary[d['class']].append(d['instance_id'])
        for cls in sorted(summary):
            ids = sorted(set(summary[cls]))
            self.quick_text.insert(tk.END,
                f"{cls}: {len(summary[cls])} obiektów (ID: {', '.join(map(str, ids))})\n")
        self.quick_text.insert(tk.END, f"\nZapisano jako: {out}\n")
        self.status_var.set(f"Analiza zakończona – wynik: {out}")

    def run(self):
        self.root.mainloop()