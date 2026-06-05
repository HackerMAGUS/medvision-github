# locales/ru.py

LABELS = {
    "person":       "человек",
    "car":          "машина",
    "bicycle":      "велосипед",
    "motorcycle":   "мотоцикл",
    "bus":          "автобус",
    "train":        "поезд",
    "truck":        "грузовик",
    "traffic light": "светофор",
    "stop sign":    "знак",
    "bench":        "скамейка",
    "bird":         "птица",
    "cat":          "кошка",
    "dog":          "собака",
    "chair":        "стул",
    "potted plant": "растение",
    "bed":          "кровать",
    "dining table": "стол",
    "tv":           "телевизор",
    "door":         "дверь",
    "pole":         "столб",
    # Генерический label от DecisionLayer (blocked-path heuristic)
    "obstacle":     "препятствие",
    "unknown":      "препятствие",
}

ZONES = {
    "left": "Слева",
    "center": "Впереди",
    "right": "Справа",
    "unknown": ""
}

DISTANCES = {
    "near": "близко",
    "mid": "",   # Игнорируем для краткости
    "far": "",   # Игнорируем для краткости
    "unknown": ""
}

def format_phrase(zone_text, label_text, distance_text):
    """
    Шаблон сборки фразы.
    Примеры:
      'Впереди препятствие, близко'
      'Слева человек'
      'Впереди машина'
    """
    phrase = f"{zone_text} {label_text}".strip()
    if distance_text:
        phrase += f", {distance_text}"
    return phrase
