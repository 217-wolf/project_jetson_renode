import time
from pathlib import Path
from benchmark.metrics.detection_metrics import (
    DetectionMetricsCalculator, load_yolo_labels, yolo_to_xyxy
)
from benchmark.metrics.class_mapping import is_person_gt, is_person_pred


from ultralytics import YOLO
import cv2

#indeksy z .yaml VisDrone
VISDRONE_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor"
}
  

class YOLOBenchmarkRunner:

    def __init__(self, model_path, confidence=0.5, image_size=640, labels_dir=None, iou_threshold=0.5):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.model_names = self.model.names
        self.confidence = confidence
        self.image_size = image_size
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.iou_threshold = iou_threshold
    


    def run(self, dataset_path):

        images = [
            p for p in sorted(Path(dataset_path).glob("*"))
            if p.suffix.lower() in [".jpg",".jpeg",".png"]
        ]

        if not images:
            raise ValueError("Brak obrazów w dataset")

        # warm-up modelu -> testy były biased bo pierwszy test jest obarczony wydłuzonym czasem przez pierwsze uruchomienie modeli
        warmup_image = cv2.imread(str(images[0]))
        
        for _ in range(5):
            self.model(
                warmup_image,
                conf=self.confidence,
                imgsz=self.image_size,
                verbose=False
            )

        total_time = 0
        total_detections = 0
        confidences = []
        latencies = []

        calculator = (
            DetectionMetricsCalculator(iou_threshold=self.iou_threshold) #wczesniej bylo 0.5
            if self.labels_dir and self.labels_dir.is_dir()
            else None
        )

        for image_path in images:

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            start = time.perf_counter()

            results = self.model(
                image,
                conf=self.confidence,
                imgsz=self.image_size,
                verbose=False
            )

            end = time.perf_counter()

            latency = end - start
            latencies.append(latency)

            total_time += end - start

            for result in results:
                boxes = result.boxes
                total_detections += len(boxes)
                for box in boxes:
                    confidences.append(float(box.conf[0]))

                if calculator is not None:
                    img_h, img_w = image.shape[:2]
                    label_path = self.labels_dir / (image_path.stem + ".txt")
                    gt_labels = load_yolo_labels(label_path)
                    # przy budowaniu gt_classes i pred_classes w pętli:
                    gt_boxes_person = []
                    for label in gt_labels:
                        if is_person_gt(label["class"], VISDRONE_NAMES):
                            gt_boxes_person.append(yolo_to_xyxy(label, img_w, img_h))

                    pred_boxes_person = []
                    pred_confs_person = []
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        if is_person_pred(cls_id, self.model_names):
                            pred_boxes_person.append(tuple(box.xyxy[0].tolist()))
                            pred_confs_person.append(float(box.conf[0]))

                    # wszystkie klasy = 0 (jedna kategoria "person") dla obu list
                    calculator.update(
                        pred_boxes_person, pred_confs_person, [0]*len(pred_boxes_person),
                        gt_boxes_person, [0]*len(gt_boxes_person),
                    )

        fps = len(images) / total_time if total_time else 0

        avg_latency = (
            sum(latencies)/len(latencies)
            if latencies else 0
        )
        #avg_latency_ms = avg_latency * 1000

        avg_detections = total_detections / len(images)

        avg_conf = (
            sum(confidences)/len(confidences)
            if confidences else 0
        )

        quality = calculator.compute() if calculator is not None else {}

        return {
            "model": self.model_path,
            "images": len(images),
            "detections": total_detections,
            "fps": round(fps,2),
            "avg_confidence": round(avg_conf,3),
            "latency_ms": round(avg_latency*1000,2),
            "avg_detections": round(avg_detections,2),
            **quality
        }