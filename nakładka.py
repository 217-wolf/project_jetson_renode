import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(34, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 64),
        )

    def forward(self, x):
        embedding = self.net(x)
        # L2 normalizacja → embedding leży na sferze jednostkowej
        # dzięki temu Cosine Similarity = dot product
        return F.normalize(embedding, p=2, dim=1)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Odległości euklidesowe (po L2 norm = to samo co cosine distance)
        dist_pos = F.pairwise_distance(anchor, positive)  # ta sama osoba
        dist_neg = F.pairwise_distance(anchor, negative)  # inna osoba

        # max(0, dist_pos - dist_neg + margin)
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()


n_buffer = 8

def normalize_skeleton(kpts):
    """
    Czyli jak daleko punkty są oddalone od centrum bioder
    """
    hip_center = (kpts[11] + kpts[12]) / 2  # lewe + prawe biodro
    kpts_centered = kpts - hip_center #ktps to cała tablica numpy (17, 2), przez to biodra są w centrum ukladu wsp

    head = kpts[0]
    scale = np.linalg.norm(head - hip_center) + 1e-6 #długość wektora + 1e-6 zeby nie dzielic przez zero
    kpts_normalized = kpts_centered / scale

    return kpts_normalized.flatten()


model = YOLO('yolo11n-pose.pt')

cap = cv2.VideoCapture(0)

embedding_buffers = defaultdict(lambda: deque(maxlen=n_buffer))


print("'q', aby wyjsc")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    """
    Z dokumentacjigit
    The persist=True argument tells the tracker that the current image or frame is the next 
    in a sequence and to expect tracks from the previous image in the current image.
    """
    results = model.track(frame, persist=True, stream=True, classes=0, tracker="botsort.yaml")

    for r in results:
        annotated_frame = r.plot()
        if r.keypoints is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            points = r.keypoints.xyn.cpu().numpy() #znormalizowane punkty, w razie jak ktos sie bedzie przyblizal do kamery, albo oddalał, to nie bedzie takiej znaczej roznicy

            for person_id, kpts in zip(ids, points):
                # kpts ma kształt (17, 2) -> 17 stawów, każdy ma x i y

                embedding = normalize_skeleton(kpts)
                embedding_buffers[person_id].append(embedding)
                stable_embedding = np.mean(embedding_buffers[person_id], axis=0) #tu srednia akurat tych klatek, moze cos innego sprytniejszego trzeba dac xd

                cv2.putText(annotated_frame, f"ID: {person_id}",(int(kpts[0][0] * frame.shape[1]), int(kpts[0][1] * frame.shape[0]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLOv11 Pose Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()