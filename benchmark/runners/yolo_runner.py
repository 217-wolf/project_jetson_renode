import time
from pathlib import Path

from ultralytics import YOLO
import cv2

class YOLOBenchmarkRunner:

    def __init__(self, model_path, confidence=0.5, image_size=640):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size
    


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

        for image_path in images:

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            start = time.time()

            results = self.model(
                image,
                conf=self.confidence,
                imgsz=self.image_size,
                verbose=False
            )

            end = time.time()

            latency = end - start
            latencies.append(latency)

            total_time += end - start

            for result in results:
                boxes = result.boxes

                total_detections += len(boxes)

                for box in boxes:
                    confidences.append(
                        float(box.conf[0])
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


        return {
            "model": self.model_path,
            "images": len(images),
            "detections": total_detections,
            "fps": round(fps,2),
            "avg_confidence": round(avg_conf,3),
            "latency_ms": round(avg_latency*1000,2),
            "avg_detections": round(avg_detections,2)
        }