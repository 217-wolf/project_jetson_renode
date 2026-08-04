from pathlib import Path


def load_yolo_labels(label_path):
    # wczytanie etykiet od YOLO
    boxes = []
    if not Path(label_path).exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = map(float, parts)
            boxes.append({"class": int(cls), "xc": xc, "yc": yc, "w": w, "h": h})
    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    xc, yc, w, h = box["xc"] * img_w, box["yc"] * img_h, box["w"] * img_w, box["h"] * img_h
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)

    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0


class DetectionMetricsCalculator:

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, pred_boxes_xyxy, pred_confidences, gt_boxes_xyxy):

        matched_gt = set()

        order = sorted(range(len(pred_boxes_xyxy)), key=lambda i: -pred_confidences[i])

        for i in order:
            pred = pred_boxes_xyxy[i]
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_boxes_xyxy):
                if gt_idx in matched_gt:
                    continue
                current_iou = iou(pred, gt)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                self.tp += 1
                matched_gt.add(best_gt_idx)
            else:
                self.fp += 1

        self.fn += len(gt_boxes_xyxy) - len(matched_gt)

    def compute(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        # mAP50 to ta miara jakości używana w sztucznej inteligencji do oceny modeli wykrywających obiekty na obrazach 
        # musi pokrywać się z prawdziwym obiektem w co najmniej 50% aby wykrycie zaliczyć jako poprawne
        map50 = precision * recall

        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "mAP50": round(map50, 3),
        }