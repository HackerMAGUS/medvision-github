import os
import urllib.request
from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODELS_DIR = Path(__file__).resolve().parent / "models"

class MergedDetectionResult:
    def __init__(self, detections):
        self.detections = detections

class Detector:
    def __init__(self, base_model_path=None, custom_model_path=None):
        self.base_model_path = str(base_model_path or MODELS_DIR / "efficientdet_lite0.tflite")
        self.custom_model_path = str(custom_model_path or MODELS_DIR / "custom_nav_model.tflite")
        self._ensure_model_exists()
        
        # 1. Base Model
        base_options = python.BaseOptions(model_asset_path=self.base_model_path)
        options1 = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.3,  # Show detections with >30% confidence
            max_results=5         # Detect up to 5 objects per frame
        )
        self.base_detector = vision.ObjectDetector.create_from_options(options1)
        
        # 2. Custom Model (optional)
        self.custom_detector = None
        if os.path.exists(self.custom_model_path):
            print(f"[Detector] Подключаем кастомную модель: {self.custom_model_path}")
            custom_base_options = python.BaseOptions(model_asset_path=self.custom_model_path)
            options2 = vision.ObjectDetectorOptions(
                base_options=custom_base_options,
                score_threshold=0.3,
                max_results=5
            )
            self.custom_detector = vision.ObjectDetector.create_from_options(options2)
        else:
            print(f"[Detector] Кастомная модель {self.custom_model_path} не найдена. Работаем только с базовой.")

    def _ensure_model_exists(self):
        if not os.path.exists(self.base_model_path):
            print(f"[Detector] Model {self.base_model_path} not found. Downloading...")
            url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
            try:
                urllib.request.urlretrieve(url, self.base_model_path)
                print("[Detector] Model downloaded successfully.")
            except Exception as e:
                print(f"[Detector] Failed to download model: {e}")
                print("[Detector] Please download it manually and place it in the project directory.")

    def detect(self, rgb_frame):
        """
        Runs object detection on the provided RGB frame.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Инференс базовой модели
        res_base = self.base_detector.detect(mp_image)
        all_detections = list(res_base.detections) if res_base.detections else []
        
        # Инференс кастомной модели (если есть)
        if self.custom_detector:
            res_custom = self.custom_detector.detect(mp_image)
            if res_custom.detections:
                all_detections.extend(res_custom.detections)
                
        return MergedDetectionResult(all_detections)
