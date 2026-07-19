from collections import Counter
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ilids_vid"
    / "i-LIDS-VID"
    / "sequences"
)

MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n-pose.pt"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "diagnostic_tests"
    / "ilids_pose_quality_v2"
)

SAMPLE_SIZE = 300
RANDOM_SEED = 42

KEYPOINT_CONFIDENCE = 0.25
MIN_VISIBLE_KEYPOINTS = 8
MIN_POINTS_INSIDE_BOX_RATIO = 0.85

MAX_SAVED_USABLE = 20
MAX_SAVED_NOT_USABLE = 30


# Połączenia punktów w szkielecie YOLO/COCO.
SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def percentage(value: int, total: int) -> float:
    if total == 0:
        return 0.0

    return 100.0 * value / total


def select_balanced_sample() -> list[Path]:
    """
    Losuje po 150 obrazów z cam1 i cam2.
    """
    rng = random.Random(RANDOM_SEED)

    cam1_images = sorted((DATA_ROOT / "cam1").rglob("*.png"))
    cam2_images = sorted((DATA_ROOT / "cam2").rglob("*.png"))

    if not cam1_images:
        raise FileNotFoundError(
            f"Nie znaleziono obrazów w: {DATA_ROOT / 'cam1'}"
        )

    if not cam2_images:
        raise FileNotFoundError(
            f"Nie znaleziono obrazów w: {DATA_ROOT / 'cam2'}"
        )

    sample_per_camera = SAMPLE_SIZE // 2

    selected = (
        rng.sample(cam1_images, sample_per_camera)
        + rng.sample(cam2_images, sample_per_camera)
    )

    rng.shuffle(selected)
    return selected


def choose_best_detection(result) -> int | None:
    """
    Wybiera wykrycie, które:
    - ma wysokie confidence,
    - jest blisko środka obrazu,
    - zajmuje znaczną część obrazu.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None

    if result.keypoints is None or len(result.keypoints) == 0:
        return None

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    box_confidences = result.boxes.conf.detach().cpu().numpy()

    image_height, image_width = result.orig_shape

    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0

    maximum_center_distance = np.hypot(
        image_center_x,
        image_center_y,
    )

    best_index = None
    best_score = -float("inf")

    for index, (box, confidence) in enumerate(
        zip(boxes, box_confidences)
    ):
        x1, y1, x2, y2 = box

        clipped_x1 = np.clip(x1, 0, image_width)
        clipped_y1 = np.clip(y1, 0, image_height)
        clipped_x2 = np.clip(x2, 0, image_width)
        clipped_y2 = np.clip(y2, 0, image_height)

        box_width = max(0.0, clipped_x2 - clipped_x1)
        box_height = max(0.0, clipped_y2 - clipped_y1)

        box_area = box_width * box_height
        image_area = image_width * image_height

        area_ratio = box_area / max(image_area, 1.0)
        area_score = min(area_ratio / 0.35, 1.0)

        box_center_x = (x1 + x2) / 2.0
        box_center_y = (y1 + y2) / 2.0

        center_distance = np.hypot(
            box_center_x - image_center_x,
            box_center_y - image_center_y,
        )

        normalized_distance = (
            center_distance / max(maximum_center_distance, 1.0)
        )

        centrality_score = max(0.0, 1.0 - normalized_distance)

        score = (
            0.50 * float(confidence)
            + 0.30 * centrality_score
            + 0.20 * area_score
        )

        if score > best_score:
            best_score = score
            best_index = index

    return best_index


def point_inside_box(
    point: np.ndarray,
    box: np.ndarray,
    margin_ratio: float = 0.08,
) -> bool:
    x1, y1, x2, y2 = box

    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)

    margin_x = margin_ratio * box_width
    margin_y = margin_ratio * box_height

    x, y = point

    return bool(
        x1 - margin_x <= x <= x2 + margin_x
        and y1 - margin_y <= y <= y2 + margin_y
    )


def validate_pose(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    box: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[bool, list[str], np.ndarray]:
    """
    Sprawdza, czy szkielet ma sens geometryczny.
    """
    reasons: list[str] = []

    image_height, image_width = image_shape
    x1, y1, x2, y2 = box

    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)

    clipped_width = max(
        min(x2, image_width) - max(x1, 0),
        0,
    )

    clipped_height = max(
        min(y2, image_height) - max(y1, 0),
        0,
    )

    box_area_ratio = (
        clipped_width
        * clipped_height
        / max(image_width * image_height, 1)
    )

    box_height_ratio = clipped_height / max(image_height, 1)

    visible = confidences >= KEYPOINT_CONFIDENCE
    visible_count = int(np.count_nonzero(visible))

    if visible_count < MIN_VISIBLE_KEYPOINTS:
        reasons.append("too_few_keypoints")

    both_shoulders = bool(visible[5] and visible[6])
    both_hips = bool(visible[11] and visible[12])

    if not both_shoulders:
        reasons.append("missing_shoulders")

    if not both_hips:
        reasons.append("missing_hips")

    if box_area_ratio < 0.15:
        reasons.append("box_too_small")

    if box_height_ratio < 0.45:
        reasons.append("box_too_short")

    visible_indices = np.flatnonzero(visible)

    if len(visible_indices) > 0:
        inside_count = sum(
            point_inside_box(keypoints[index], box)
            for index in visible_indices
        )

        inside_ratio = inside_count / len(visible_indices)

        if inside_ratio < MIN_POINTS_INSIDE_BOX_RATIO:
            reasons.append("points_outside_box")

    # Sprawdzenie położenia barków i bioder.
    if both_shoulders and both_hips:
        shoulder_center = np.mean(keypoints[[5, 6]], axis=0)
        hip_center = np.mean(keypoints[[11, 12]], axis=0)

        torso_height = hip_center[1] - shoulder_center[1]

        if torso_height < 0.03 * box_height:
            reasons.append("hips_not_below_shoulders")

        if torso_height > 0.60 * box_height:
            reasons.append("torso_too_long")

    vertical_tolerance = 0.08 * box_height

    # Lewa noga: biodro -> kolano -> kostka.
    for hip_index, knee_index, ankle_index, side_name in [
        (11, 13, 15, "left"),
        (12, 14, 16, "right"),
    ]:
        if visible[hip_index] and visible[knee_index]:
            if (
                keypoints[knee_index, 1]
                < keypoints[hip_index, 1] - vertical_tolerance
            ):
                reasons.append(f"{side_name}_knee_above_hip")

        if visible[knee_index] and visible[ankle_index]:
            if (
                keypoints[ankle_index, 1]
                < keypoints[knee_index, 1] - vertical_tolerance
            ):
                reasons.append(f"{side_name}_ankle_above_knee")

    # Odrzucenie absurdalnie długich odcinków.
    body_segments = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (5, 11),
        (6, 12),
    ]

    for start_index, end_index in body_segments:
        if not (visible[start_index] and visible[end_index]):
            continue

        segment_length = np.linalg.norm(
            keypoints[start_index] - keypoints[end_index]
        )

        if segment_length > 0.75 * box_height:
            reasons.append("segment_too_long")
            break

    # Usuwamy duplikaty, ale zachowujemy kolejność.
    unique_reasons = list(dict.fromkeys(reasons))

    return len(unique_reasons) == 0, unique_reasons, visible


def draw_selected_pose(
    image: np.ndarray,
    box: np.ndarray,
    keypoints: np.ndarray,
    visible: np.ndarray,
    confidence: float,
    reasons: list[str],
) -> np.ndarray:
    """
    Rysuje tylko wybrane wykrycie zamiast wszystkich wyników YOLO.
    """
    output = image.copy()

    x1, y1, x2, y2 = box.astype(int)

    is_usable = len(reasons) == 0

    box_color = (0, 220, 0) if is_usable else (0, 0, 255)
    skeleton_color = (255, 160, 0)
    point_color = (255, 0, 255)

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        box_color,
        1,
    )

    for start_index, end_index in SKELETON_CONNECTIONS:
        if visible[start_index] and visible[end_index]:
            start_point = tuple(
                keypoints[start_index].astype(int)
            )

            end_point = tuple(
                keypoints[end_index].astype(int)
            )

            cv2.line(
                output,
                start_point,
                end_point,
                skeleton_color,
                1,
            )

    for index, point in enumerate(keypoints):
        if not visible[index]:
            continue

        cv2.circle(
            output,
            tuple(point.astype(int)),
            2,
            point_color,
            -1,
        )

    status = "USABLE" if is_usable else "NOT USABLE"

    cv2.putText(
        output,
        f"{status} conf={confidence:.2f}",
        (2, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.25,
        box_color,
        1,
        cv2.LINE_AA,
    )

    # Powiększamy wynik czterokrotnie, żeby był czytelny.
    output = cv2.resize(
        output,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_NEAREST,
    )

    if reasons:
        reason_text = ", ".join(reasons[:3])

        cv2.putText(
            output,
            reason_text,
            (5, output.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {MODEL_PATH}"
        )

    selected_images = select_balanced_sample()

    # Usuwamy poprzednie wyniki wersji 2, żeby się nie mieszały.
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    usable_directory = OUTPUT_ROOT / "usable"
    not_usable_directory = OUTPUT_ROOT / "not_usable"

    usable_directory.mkdir(parents=True, exist_ok=True)
    not_usable_directory.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    stats = {
        "tested": 0,
        "person_detected": 0,
        "usable": 0,
        "not_usable": 0,
        "visible_keypoints_sum": 0,
    }

    failure_reasons = Counter()

    saved_usable = 0
    saved_not_usable = 0

    print(f"Liczba testowanych obrazów: {len(selected_images)}")
    print(f"Model: {MODEL_PATH}")
    print("Uruchamianie poprawionego testu YOLO Pose...")

    results = model.predict(
        source=[str(path) for path in selected_images],
        device=0,
        imgsz=256,
        conf=0.10,
        batch=32,
        stream=True,
        verbose=False,
    )

    for image_path, result in zip(selected_images, results):
        stats["tested"] += 1

        image = cv2.imread(str(image_path))

        if image is None:
            failure_reasons["cannot_read_image"] += 1
            stats["not_usable"] += 1
            continue

        best_index = choose_best_detection(result)

        if best_index is None:
            failure_reasons["no_detection"] += 1
            stats["not_usable"] += 1

            if saved_not_usable < MAX_SAVED_NOT_USABLE:
                enlarged = cv2.resize(
                    image,
                    None,
                    fx=4,
                    fy=4,
                    interpolation=cv2.INTER_NEAREST,
                )

                output_path = (
                    not_usable_directory
                    / f"no_detection_{saved_not_usable:03d}_{image_path.name}"
                )

                cv2.imwrite(str(output_path), enlarged)
                saved_not_usable += 1

            continue

        stats["person_detected"] += 1

        keypoint_confidences = result.keypoints.conf

        if keypoint_confidences is None:
            failure_reasons["no_keypoint_confidence"] += 1
            stats["not_usable"] += 1
            continue

        keypoints = (
            result.keypoints.xy[best_index]
            .detach()
            .cpu()
            .numpy()
        )

        confidences = (
            keypoint_confidences[best_index]
            .detach()
            .cpu()
            .numpy()
        )

        box = (
            result.boxes.xyxy[best_index]
            .detach()
            .cpu()
            .numpy()
        )

        detection_confidence = float(
            result.boxes.conf[best_index].item()
        )

        is_usable, reasons, visible = validate_pose(
            keypoints=keypoints,
            confidences=confidences,
            box=box,
            image_shape=image.shape[:2],
        )

        stats["visible_keypoints_sum"] += int(
            np.count_nonzero(visible)
        )

        annotated = draw_selected_pose(
            image=image,
            box=box,
            keypoints=keypoints,
            visible=visible,
            confidence=detection_confidence,
            reasons=reasons,
        )

        if is_usable:
            stats["usable"] += 1

            if saved_usable < MAX_SAVED_USABLE:
                output_path = (
                    usable_directory
                    / f"usable_{saved_usable:03d}_{image_path.name}"
                )

                cv2.imwrite(str(output_path), annotated)
                saved_usable += 1
        else:
            stats["not_usable"] += 1
            failure_reasons.update(reasons)

            if saved_not_usable < MAX_SAVED_NOT_USABLE:
                reason_name = reasons[0] if reasons else "unknown"

                output_path = (
                    not_usable_directory
                    / (
                        f"{reason_name}_"
                        f"{saved_not_usable:03d}_"
                        f"{image_path.name}"
                    )
                )

                cv2.imwrite(str(output_path), annotated)
                saved_not_usable += 1

    tested = stats["tested"]
    detected = stats["person_detected"]

    average_visible = (
        stats["visible_keypoints_sum"] / detected
        if detected > 0
        else 0.0
    )

    reasons_text = "\n".join(
        f"{reason:30s} {count:4d}"
        for reason, count in failure_reasons.most_common()
    )

    if not reasons_text:
        reasons_text = "Brak odrzuconych klatek."

    report = f"""
===== RAPORT YOLO POSE V2 — iLIDS-VID =====

Przetestowane obrazy:       {tested}

Wykryta osoba:              {stats["person_detected"]}
Procent wykryć:             {percentage(stats["person_detected"], tested):.2f}%

Klatki użyteczne:           {stats["usable"]}
Procent użytecznych:        {percentage(stats["usable"], tested):.2f}%

Klatki odrzucone:           {stats["not_usable"]}
Procent odrzuconych:        {percentage(stats["not_usable"], tested):.2f}%

Średnia liczba widocznych
punktów po wykryciu:        {average_visible:.2f} / 17

Próg punktu:                {KEYPOINT_CONFIDENCE}
Minimum punktów:            {MIN_VISIBLE_KEYPOINTS}
Wymagane punkty w bbox:     {MIN_POINTS_INSIDE_BOX_RATIO:.2f}

===== POWODY ODRZUCENIA =====

{reasons_text}
"""

    print(report)

    report_path = OUTPUT_ROOT / "report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(f"Raport zapisano w: {report_path}")
    print(f"Poprawne przykłady: {usable_directory}")
    print(f"Odrzucone przykłady: {not_usable_directory}")


if __name__ == "__main__":
    main()