#!/usr/bin/env python3
import cv2
import numpy as np
import torch
import sys
from scipy.spatial.distance import cosine
from ultralytics import YOLO

# ===================== KONFIGURACJA =====================
TARGET_CLASS = 'person'          # 'person' lub 'bus'
SIMILARITY_THRESHOLD = 0.85 if TARGET_CLASS == 'bus' else 0.75
CAMERA_ID = 0
USE_CUDA = True
SHOW_GUI = True               # False, jeśli nie masz monitora
# ==========================================================

def load_embedding_model():
    """Ładuje model embeddingowy w zależności od TARGET_CLASS."""
    device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')
    if TARGET_CLASS == 'person':
        try:
            import torchreid
        except ImportError:
            sys.exit("Potrzebny torchreid: pip install torchreid")
        model = torchreid.models.build_model(
            name='osnet_x0_25', num_classes=1000, pretrained=True
        ).to(device).eval()
    elif TARGET_CLASS == 'bus':
        import torchvision.models as models
        efficientnet = models.efficientnet_b0(pretrained=True)
        model = torch.nn.Sequential(*list(efficientnet.children())[:-1]).to(device).eval()
    else:
        sys.exit("Nieznany TARGET_CLASS")
    return model, device

def get_embedding(image_crop, model, device):
    """Wyciąga embedding z wycinka obrazu."""
    if image_crop.size == 0:
        return None
    if TARGET_CLASS == 'person':
        img = cv2.resize(image_crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
    else:  # obiekty
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
        features = model(img_tensor)
    return features.flatten().cpu().numpy()

def compare_embeddings(emb1, emb2):
    """Podobieństwo cosinusowe (1 = identyczne)."""
    return 1 - cosine(emb1.flatten(), emb2.flatten())

def main():
    # 1. YOLO
    model_yolo = YOLO('yolo26n.pt')  # upewnij się, że plik istnieje
    if USE_CUDA and torch.cuda.is_available():
        model_yolo.to('cuda')
        print("YOLO na GPU (CUDA)")
    else:
        print("YOLO na CPU")

    # 2. Model embeddingowy
    embedding_model, device = load_embedding_model()
    print(f"Embedding model: {TARGET_CLASS} na {device}")

    # 3. Wczytaj wzorzec
    ref_file = f"wzorzec_{TARGET_CLASS}.npy"
    try:
        ref_emb = np.load(ref_file)
        print(f"Wczytano wzorzec z {ref_file}")
    except FileNotFoundError:
        sys.exit(f"Brak pliku {ref_file}. Najpierw wygeneruj go skryptem do zdjęć.")

    # 4. Klasy YOLO
    CLASS_IDS = [0] if TARGET_CLASS == 'person' else [5]

    # 5. Kamera
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Kamera uruchomiona. Naciśnij 'q', aby wyjść.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detekcja
        results = model_yolo(frame, classes=CLASS_IDS, conf=0.25, imgsz=640, verbose=False)

        # Przetwarzanie wykrytych obiektów
        annotated_frame = frame.copy()
        detected_info = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    crop = frame[y1:y2, x1:x2]
                    emb = get_embedding(crop, embedding_model, device)
                    if emb is not None:
                        sim = compare_embeddings(ref_emb, emb)
                        detected_info.append({
                            'bbox': (x1, y1, x2, y2),
                            'similarity': sim
                        })

        # Rysowanie ramek
        for info in detected_info:
            x1, y1, x2, y2 = info['bbox']
            sim = info['similarity']
            if sim >= SIMILARITY_THRESHOLD:
                color = (0, 255, 0)          # zielony – TO TEN SAM
                label = f"MATCH {sim:.2f}"
            else:
                color = (0, 0, 255)          # czerwony – INNY
                label = f"OTHER {sim:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Konsola
        if detected_info:
            status = " | ".join([f"sim={d['similarity']:.2f}" for d in detected_info])
            print(f"Obiekty: {len(detected_info)}  [{status}]")

        if SHOW_GUI:
            cv2.imshow("Rozpoznawanie obiektu", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Jeśli GUI wyłączone, można zatrzymać przez Ctrl+C
            pass

    cap.release()
    if SHOW_GUI:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()