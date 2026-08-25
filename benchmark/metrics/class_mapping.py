# Standardowe nazwy klas w VisDrone (sprawdź swój dataset i podmień jeśli inne)
VISDRONE_PERSON_CLASSES = {"pedestrian", "people"}

# COCO - model YOLO
COCO_PERSON_CLASS_NAME = "person"


def is_person_gt(class_id, gt_names_map):
    """gt_names_map: dict {id: name} z pliku data.yaml datasetu"""
    return gt_names_map.get(class_id) in VISDRONE_PERSON_CLASSES


def is_person_pred(class_id, model_names_map):
    """model_names_map: model.names z ultralytics"""
    return model_names_map.get(class_id) == COCO_PERSON_CLASS_NAME