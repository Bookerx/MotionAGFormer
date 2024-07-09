import numpy as np
from ultralytics import YOLO


def expand_bbox(bbox, scale=1.2):
    """
    Expand the bounding box by a given scale factor.

    Parameters:
    bbox (tuple): A tuple (x_min, y_min, x_max, y_max) representing the bounding box.
    scale (float): The scale factor to expand the bounding box. Default is 1.2 (20% expansion).

    Returns:
    tuple: A tuple (new_x_min, new_y_min, new_x_max, new_y_max) representing the expanded bounding box.
    """
    x_min, y_min, x_max, y_max = bbox

    # Calculate the width and height of the original bbox
    width = x_max - x_min
    height = y_max - y_min

    # Calculate the amount to expand
    width_expand = (scale - 1) * width / 2
    height_expand = (scale - 1) * height / 2

    # Calculate the new bbox coordinates
    new_x_min = x_min - width_expand
    new_y_min = y_min - height_expand
    new_x_max = x_max + width_expand
    new_y_max = y_max + height_expand

    return (new_x_min, new_y_min, new_x_max, new_y_max)


def detect_persons(image_path, confidence=0.7):
    # Load YOLOv8 model
    model = YOLO('lib/yolov8/yolov8x.pt')

    results = model(image_path, conf=confidence)

    bboxes = []
    scores = []

    for result in results:
        for detection in result.boxes:
            if detection.cls == 0:  # class 0 is 'person' in COCO dataset
                bbox = detection.xyxy[0].tolist()  # Convert to list
                score = detection.conf[0].item()  # Convert to float
                if score >= confidence:  # Apply confidence threshold
                    bboxes.append(expand_bbox(bbox))
                    scores.append(score)

    return np.array(bboxes), np.array(scores)
