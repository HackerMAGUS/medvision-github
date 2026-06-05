"""
Скрипт для дообучения легкой модели EfficientDet-Lite0 на ваших 3 классах.
Скрипт оптимизирован для запуска на локальном ПК или в Google Colab.

Установка зависимостей:
pip install mediapipe-model-maker tensorflow
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

print("Проверка библиотек...")
try:
    import tensorflow as tf
    from mediapipe_model_maker import object_detector
except ImportError:
    print("ОШИБКА: Установите библиотеки:")
    print("pip install mediapipe-model-maker tensorflow")
    exit(1)

# ==========================================
# 1. КОНФИГУРАЦИЯ ДАТАСЕТА
# ==========================================
# Датасет должен быть в формате COCO.
# Структура:
# custom_dataset/
#   train/
#     images/
#     labels.json
#   val/
#     images/
#     labels.json

TRAIN_DATASET_PATH = str(BASE_DIR / "custom_dataset" / "train")
VAL_DATASET_PATH = str(BASE_DIR / "custom_dataset" / "val")

EXPORT_DIR = str(BASE_DIR / "exported_model")
EPOCHS = 30
BATCH_SIZE = 8

def main():
    if not os.path.exists(TRAIN_DATASET_PATH) or not os.path.exists(VAL_DATASET_PATH):
        print(f"[!] Внимание: Папки {TRAIN_DATASET_PATH} или {VAL_DATASET_PATH} не найдены!")
        print("Положите ваш датасет в папку 'custom_dataset' перед запуском.")
        return

    print("=== [1/4] Загрузка Датасета ===")
    train_data = object_detector.Dataset.from_coco_folder(TRAIN_DATASET_PATH, cache_dir="/tmp/od_data/train")
    val_data = object_detector.Dataset.from_coco_folder(VAL_DATASET_PATH, cache_dir="/tmp/od_data/val")

    print("=== [2/4] Подготовка Backbone модели (EfficientDet-Lite0) ===")
    # Выбираем архитектуру, идеальную для телефона
    spec = object_detector.SupportedModels.EFFICIENTDET_LITE0
    
    hparams = object_detector.HParams(export_dir=EXPORT_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE)
    options = object_detector.ObjectDetectorOptions(
        supported_model=spec,
        hparams=hparams
    )

    print("=== [3/4] Старт Transfer Learning (Fine-Tuning) ===")
    print(f"Обучение начнется на {EPOCHS} эпох...")
    model = object_detector.ObjectDetector.create(
        train_data=train_data,
        validation_data=val_data,
        options=options
    )

    print("=== [4/4] Оценка модели и Экспорт ===")
    loss, coco_metrics = model.evaluate(val_data)
    print(f"Validation loss: {loss}")
    
    model.export_model("custom_nav_model.tflite")
    print(f"✅ УСПЕХ! Готовая on-device модель сохранена: {EXPORT_DIR}/custom_nav_model.tflite")
    print(f"Эта модель весит ~4.5 МБ и имеет вшитую Metadata. Просто положите её рядом с main.py.")

if __name__ == "__main__":
    main()
