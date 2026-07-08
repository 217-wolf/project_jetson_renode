import cv2
import numpy as np
import torch
import sys
from ultralytics import YOLO
from scipy.spatial.distance import cosine

# ==============================================
# KONFIGURACJA
# ==============================================
TARGET_CLASS = 'person'  # zmień na 'person' dla osób
REFERENCE_IMAGE = "idol.jpeg"   # zdjęcie wzorcowe
TEST_IMAGE = "idol_uczy.jpeg"             # zdjęcie testowe

# Próg podobieństwa (dostosuj)
if TARGET_CLASS == 'person':
    SIMILARITY_THRESHOLD = 0.75
else:
    SIMILARITY_THRESHOLD = 0.85

# Czy pokazać obrazek na ekranie (False na Jetson bez GUI)
SHOW_IMAGE = False  # zmień na True, jeśli masz podłączony monitor i działa cv2.imshow

# ==============================================
# INICJALIZACJA
# ==============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Urządzenie: {device}")

# YOLO
detector = YOLO('yolov8n.pt')
detector.to(device)

if TARGET_CLASS == 'person':
    CLASS_IDS = [0]
    import torchreid
    embedding_model = torchreid.models.build_model(
        name='osnet_x0_25', num_classes=1000, pretrained=True
    ).to(device).eval()
    print("Model: OSNet (osoby)")
elif TARGET_CLASS == 'bus':
    CLASS_IDS = [5]
    import torchvision.models as models
    efficientnet = models.efficientnet_b0(pretrained=True)
    embedding_model = torch.nn.Sequential(*list(efficientnet.children())[:-1]).to(device).eval()
    print("Model: EfficientNet-B0 (obiekty)")
else:
    sys.exit("Nieznany TARGET_CLASS")

# ==============================================
# FUNKCJE
# ==============================================
def get_embedding(image_crop):
    if image_crop.size == 0:
        return None
    if TARGET_CLASS == 'person':
        img = cv2.resize(image_crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
    else:  # bus / obiekty
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
        features = embedding_model(img_tensor)
    return features.flatten().cpu().numpy()

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nie można wczytać: {image_path}")
    print(f"\nWczytano: {image_path} ({img.shape[1]}x{img.shape[0]})")
    results = detector(img, classes=CLASS_IDS, conf=0.25, imgsz=640, verbose=False)
    embeddings = []
    boxes = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                crop = img[y1:y2, x1:x2]
                emb = get_embedding(crop)
                if emb is not None:
                    embeddings.append(emb)
                    boxes.append({'bbox': (x1, y1, x2, y2), 'conf': conf})
                    print(f"  Obiekt: pewność {conf:.2f}, bbox ({x1},{y1})-({x2},{y2})")
        else:
            print("  Brak detekcji")
    return img, embeddings, boxes   # zwracamy też obraz, by móc go potem modyfikować

def compare_embeddings(emb1, emb2):
    return 1 - cosine(emb1.flatten(), emb2.flatten())

def draw_results(img, boxes, similarities, threshold):
    """Rysuje ramki: zielona + 'MATCH' jeśli sim >= threshold, inaczej czerwona + 'NO MATCH'."""
    for i, box_info in enumerate(boxes):
        x1, y1, x2, y2 = box_info['bbox']
        sim = similarities[i]
        if sim >= threshold:
            color = (0, 255, 0)       # zielony
            label = f"MATCH ({sim:.2f})"
        else:
            color = (0, 0, 255)       # czerwony
            label = f"NO MATCH ({sim:.2f})"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def save_or_show(img, filename="wynik.jpg"):
    cv2.imwrite(filename, img)
    print(f"Zapisano obraz wynikowy jako {filename}")
    if SHOW_IMAGE:
        cv2.imshow("Wynik", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ==============================================
# GŁÓWNY PROGRAM
# ==============================================
if __name__ == "__main__":
    # 1. Wzorzec
    ref_img, ref_embs, _ = process_image(REFERENCE_IMAGE)
    if len(ref_embs) == 0:
        sys.exit("Nie wykryto obiektu na obrazie referencyjnym.")
    ref_emb = ref_embs[0]
    np.save(f"wzorzec_{TARGET_CLASS}.npy", ref_emb)
    print("Zapisano embedding wzorcowy.")

    # 2. Obraz testowy
    test_img, test_embs, test_boxes = process_image(TEST_IMAGE)
    if len(test_embs) == 0:
        sys.exit("Nie wykryto obiektu na obrazie testowym.")

    # 3. Porównanie i rysowanie
    similarities = [compare_embeddings(ref_emb, emb) for emb in test_embs]
    for i, sim in enumerate(similarities):
        print(f"Obiekt {i}: podobieństwo = {sim:.4f} -> {'MATCH' if sim >= SIMILARITY_THRESHOLD else 'no match'}")

    annotated_img = draw_results(test_img.copy(), test_boxes, similarities, SIMILARITY_THRESHOLD)
    save_or_show(annotated_img)