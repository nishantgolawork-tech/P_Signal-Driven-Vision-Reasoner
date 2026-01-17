from ultralytics import YOLO

# Load YOLO model once (global)
yolo_model = YOLO("yolov8s.pt")  # small model, fast

def detect_objects(image_path, conf_threshold=0.4):
    results = yolo_model(image_path)[0]  # run inference

    detected_labels = []

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls[0])
        label = results.names[cls_id]

        detected_labels.append(label)

    return detected_labels

