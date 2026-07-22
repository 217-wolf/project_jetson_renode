import cv2
import torch
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque
from src.skeleton_utils import normalize_skeleton
from src.model import EmbeddingNet
from src.gallery import Gallery, transform_batch_emb


"""
n_buffers działa jak okno przesuwne -> ostatnie 8 katek osoby
"""
#Parametry do wpisania
emb_model_weights_path = 'embedding_net_best.pt'
similarity_margin = 0.7
n_buffers = 8

emb_buffers_dict = defaultdict(lambda: deque(maxlen=n_buffers)) #przechowuje n_buffers klatek
assigned_labels = {} #np.  tracker_id: 'osoba1' czyli kluczem jest tracker_id, a wartością jest klucz z self.known_emb z klasy Gallery


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Re-ID: urządzenie = {device}")

gallery = Gallery(5)
emb_model = EmbeddingNet().to(device)
emb_model.load_state_dict(torch.load(emb_model_weights_path, map_location=device))
emb_model.eval()

yolo_model = YOLO('yolo11n-pose.pt')
cap = cv2.VideoCapture(0)

print("'q', aby wyjsc")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    #z dokumentacji Handheld, drone, or moving-camera footage? → BoT-SORT (default; adds camera-motion compensation and optional ReID).
    results = yolo_model.track(frame, stream=True, persist=True, classes=0, tracker="botsort.yaml") #classses = 0 w COCO to ludzie
    annotated_frame = frame
    for r in results:
        annotated_frame = r.plot()
        if r.keypoints is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            points = r.keypoints.xyn.cpu().numpy()
            for tracker_id, kpts in zip(ids, points):
                emb = normalize_skeleton(kpts)
                emb_buffers_dict[tracker_id].append(emb)

                if tracker_id in assigned_labels:
                    person_id = assigned_labels[tracker_id]
                elif len(emb_buffers_dict[tracker_id]) < n_buffers: #dopóki nie będzie 8 klatek, pokazujemy surowe tracker_id z yolo
                    person_id = f'...{tracker_id}'
                else:
                    calculated_emb = transform_batch_emb(emb_buffers_dict[tracker_id], device, emb_model)
                    label = gallery.match(calculated_emb, similarity_margin)
                    assigned_labels[tracker_id] = label
                    print(f"tracker_id {tracker_id} -> {label}")

                cv2.putText(annotated_frame, person_id,(int(kpts[0][0] * frame.shape[1]), int(kpts[0][1] * frame.shape[0]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLOv11 Pose Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()