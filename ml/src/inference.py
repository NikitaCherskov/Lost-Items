from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path: str, target_classes: list[str] = None):
        self.model = YOLO(model_path)
        if target_classes:
            self.target_classes = [cls.lower() for cls in target_classes]
        else:
            self.target_classes = [cls.lower() for cls in self.model.names.values()]

    def predict(self, image_path: str):
        results = self.model(image_path)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id].lower()
                if cls_name not in self.target_classes:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
        return detections

    def visualize(self, image_path: str, detections: list[dict]):
        img = cv2.imread(str(image_path))
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_name = det["class"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img