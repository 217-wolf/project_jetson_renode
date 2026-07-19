from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class PoseDetection:
    """
    Wynik wykrycia jednej, głównej osoby.
    """

    bbox: np.ndarray
    keypoints: np.ndarray
    detection_confidence: float


class PoseDetector:
    """
    Detektor wybierający jedną główną osobę z obrazu.

    W iLIDS-VID obrazy są już przycięte wokół konkretnej osoby,
    dlatego preferujemy wykrycie:
    - z wysokim confidence,
    - blisko środka obrazu,
    - zajmujące dużą część obrazu.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: int | str = 0,
        image_size: int = 256,
        detection_threshold: float = 0.10,
    ) -> None:
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono modelu YOLO: {model_path}"
            )

        self.model = YOLO(str(model_path))
        self.device = device
        self.image_size = image_size
        self.detection_threshold = detection_threshold

    @staticmethod
    def _choose_primary_detection(result) -> int | None:
        if result.boxes is None or len(result.boxes) == 0:
            return None

        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        if result.keypoints.conf is None:
            return None

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()

        image_height, image_width = result.orig_shape

        image_center = np.array(
            [image_width / 2.0, image_height / 2.0],
            dtype=np.float32,
        )

        maximum_distance = float(
            np.linalg.norm(image_center)
        )

        image_area = float(
            max(image_width * image_height, 1)
        )

        best_index: int | None = None
        best_score = -float("inf")

        for index, (box, confidence) in enumerate(
            zip(boxes, confidences)
        ):
            x1, y1, x2, y2 = box

            clipped_x1 = np.clip(x1, 0, image_width)
            clipped_y1 = np.clip(y1, 0, image_height)
            clipped_x2 = np.clip(x2, 0, image_width)
            clipped_y2 = np.clip(y2, 0, image_height)

            box_width = max(
                float(clipped_x2 - clipped_x1),
                0.0,
            )

            box_height = max(
                float(clipped_y2 - clipped_y1),
                0.0,
            )

            area_ratio = (
                box_width * box_height
                / image_area
            )

            area_score = min(area_ratio / 0.35, 1.0)

            box_center = np.array(
                [
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                ],
                dtype=np.float32,
            )

            center_distance = float(
                np.linalg.norm(box_center - image_center)
            )

            centrality_score = max(
                0.0,
                1.0
                - center_distance
                / max(maximum_distance, 1.0),
            )

            score = (
                0.55 * float(confidence)
                + 0.25 * centrality_score
                + 0.20 * area_score
            )

            if score > best_score:
                best_score = score
                best_index = index

        return best_index

    @staticmethod
    def _convert_result(
        result,
    ) -> PoseDetection | None:
        detection_index = (
            PoseDetector._choose_primary_detection(result)
        )

        if detection_index is None:
            return None

        keypoint_xy = (
            result.keypoints.xy[detection_index]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        keypoint_confidence = (
            result.keypoints.conf[detection_index]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if keypoint_xy.shape != (17, 2):
            return None

        if keypoint_confidence.shape != (17,):
            return None

        keypoints = np.concatenate(
            [
                keypoint_xy,
                keypoint_confidence[:, None],
            ],
            axis=1,
        )

        bbox = (
            result.boxes.xyxy[detection_index]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        detection_confidence = float(
            result.boxes.conf[detection_index].item()
        )

        return PoseDetection(
            bbox=bbox,
            keypoints=keypoints,
            detection_confidence=detection_confidence,
        )

    def predict_paths(
        self,
        image_paths: Sequence[str | Path],
        batch_size: int = 32,
    ) -> Iterator[PoseDetection | None]:
        paths = [str(Path(path)) for path in image_paths]

        results = self.model.predict(
            source=paths,
            device=self.device,
            imgsz=self.image_size,
            conf=self.detection_threshold,
            batch=batch_size,
            stream=True,
            verbose=False,
        )

        for result in results:
            yield self._convert_result(result)