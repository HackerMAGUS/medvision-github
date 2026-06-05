import cv2
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODELS_DIR = Path(__file__).resolve().parent / "models"

class TrafficLightDetector:
    def __init__(self, model_path=None):
        """
        [PLACEHOLDER INTERFACE]
        Для прототипа переиспользуем MediaPipe EfficientDet.
        В будущем здесь будет загружаться кастомная YOLO/ONNX модель (traffic_light_vehicle, traffic_light_pedestrian).
        """
        model_path = str(model_path or MODELS_DIR / "efficientdet_lite0.tflite")
        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.ObjectDetectorOptions(base_options=self.base_options, score_threshold=0.25)
        self.detector = vision.ObjectDetector.create_from_options(self.options)

    def detect(self, rgb_frame, roi_mode=False):
        orig_h, orig_w = rgb_frame.shape[:2]

        if roi_mode:
            # Берём только верхние 60% кадра.
            # MediaPipe будет масштабировать именно этот обрезок до своего resolution.
            # После детекции координаты нужно спроецировать обратно на оригинал.
            roi_h = int(orig_h * 0.6)
            frame_for_detection = rgb_frame[:roi_h, :]
        else:
            frame_for_detection = rgb_frame

        # Конвертация в формат MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_for_detection)
        detection_result = self.detector.detect(mp_image)

        det_h = frame_for_detection.shape[0]
        det_w = frame_for_detection.shape[1]

        lights = []
        for d in detection_result.detections:
            cat = d.categories[0]
            # Ищем только светофоры
            if cat.category_name == "traffic light":
                bbox = d.bounding_box
                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                w = int(bbox.width)
                h = int(bbox.height)

                # Если мы передавали обрезанный сверху кадр (roi_mode=True),
                # то координаты x, y, w, h, которые вернул MediaPipe, УЖЕ 
                # соответствуют оригинальным координатам Y, так как обрезка была от Y=0.
                # Никакого искажения масштаба делать НЕЛЬЗЯ.

                # --- ЭВРИСТИКА-ЗАГЛУШКА ДО ОБУЧЕНИЯ МОДЕЛИ ---
                # Обычный светофор вытянут по вертикали (h/w > 1.8)
                # Пешеходный (или одиночная секция) часто более квадратный
                tl_type = "vehicle"
                if h > 0 and w > 0 and (h / w) < 1.8:
                    tl_type = "pedestrian"

                lights.append({
                    "bbox": (x, y, w, h),
                    "type": tl_type,
                    "confidence": cat.score
                })
        return lights
