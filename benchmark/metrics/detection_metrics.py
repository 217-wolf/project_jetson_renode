from pathlib import Path


def load_yolo_labels(label_path):
    """Load YOLO labels: class, x_center, y_center, width, height."""
    boxes = []

    if not Path(label_path).exists():
        return boxes

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, xc, yc, w, h = map(float, parts)

            boxes.append({
                "class": int(cls),
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h,
            })

    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    """Convert normalized YOLO box to pixel XYXY coordinates."""
    xc = box["xc"] * img_w
    yc = box["yc"] * img_h
    w = box["w"] * img_w
    h = box["h"] * img_h

    return (
        xc - w / 2,
        yc - h / 2,
        xc + w / 2,
        yc + h / 2,
    )


def iou(box_a, box_b):
    """Calculate Intersection over Union between two XYXY boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = (
        max(0, inter_x2 - inter_x1)
        * max(0, inter_y2 - inter_y1)
    )

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)

    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0


class DetectionMetricsCalculator:

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

        self.tp = 0
        self.fp = 0
        self.fn = 0

        self.all_predictions = []
        self.total_gt = 0

    def update(
        self,
        pred_boxes,
        pred_confidences,
        pred_classes,
        gt_boxes,
        gt_classes,
        ):
        """
        Update detection metrics for one image.

        Predictions and ground truth are matched only when:
        1. classes are equal
        2. IoU >= threshold
        3. ground-truth object has not already been matched
        """

        matched_gt = set()

        # Higher-confidence predictions are matched first.
        order = sorted(
            range(len(pred_boxes)),
            key=lambda i: -pred_confidences[i],
        )

        for pred_idx in order:

            pred_box = pred_boxes[pred_idx]
            pred_class = pred_classes[pred_idx]
            pred_conf = pred_confidences[pred_idx]

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):

                if gt_idx in matched_gt:
                    continue

                # Class must match.
                if pred_class != gt_classes[gt_idx]:
                    continue

                current_iou = iou(pred_box, gt_box)

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = gt_idx

            is_correct = best_iou >= self.iou_threshold and best_gt_idx != -1

            if is_correct:
                self.tp += 1
                matched_gt.add(best_gt_idx)
            else:
                self.fp += 1

            # Zapisujemy KAŻDĄ predykcję do liczenia mAP
            self.all_predictions.append((pred_conf, is_correct))

        self.fn += len(gt_boxes) - len(matched_gt)
        self.total_gt += len(gt_boxes)

    def compute_ap50(self):
        """ Average Precision przy IoU=0.5, metoda VOC 2010 style AP interpolation """

        if not self.all_predictions or self.total_gt == 0:
            return 0.0

        # Sortuj wszystkie predykcje (ze WSZYSTKICH obrazów) po confidence malejąco
        preds_sorted = sorted(self.all_predictions, key=lambda x: -x[0])

        tp_cumsum = 0
        fp_cumsum = 0

        precisions = []
        recalls = []

        for conf, is_correct in preds_sorted:
            if is_correct:
                tp_cumsum += 1
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / self.total_gt

            precisions.append(precision)
            recalls.append(recall)

        # All-point interpolation: dla każdego poziomu recall bierzemy
        # maksymalne precision osiągnięte przy recall >= tym poziomie
        # (standard używany w COCO/PASCAL VOC)
        interpolated_precisions = []
        max_precision_so_far = 0

        for p in reversed(precisions):
            max_precision_so_far = max(max_precision_so_far, p)
            interpolated_precisions.insert(0, max_precision_so_far)

        # AP = pole pod krzywą (całka numeryczna po zmianach recall)
        ap = 0.0
        prev_recall = 0.0

        for p, r in zip(interpolated_precisions, recalls):
            ap += p * (r - prev_recall)
            prev_recall = r

        return ap


    def compute(self):

        precision = (
            self.tp / (self.tp + self.fp)
            if (self.tp + self.fp) > 0
            else 0
        )

        recall = (
            self.tp / (self.tp + self.fn)
            if (self.tp + self.fn) > 0
            else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        ap50 = self.compute_ap50()

        print(
                    f"TP={self.tp}, "
                    f"FP={self.fp}, "
                    f"FN={self.fn}, "
                    f"mAP50={ap50:.3f}"
        )

        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "mAP50": round(ap50, 3),
        }