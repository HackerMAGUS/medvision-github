# locales/en.py

LABELS = {
    "person": "person",
    "car": "car",
    "bicycle": "bicycle",
    "traffic light": "traffic light",
    "unknown": "obstacle"
}

ZONES = {
    "left": "On the left",
    "center": "In the center",
    "right": "On the right",
    "unknown": "Somewhere"
}

DISTANCES = {
    "near": "near",
    "mid": "mid-distance",
    "far": "far",
    "unknown": ""
}

def format_phrase(zone_text, label_text, distance_text):
    phrase = f"{zone_text} {label_text}"
    if distance_text:
        phrase += f", {distance_text}"
    return phrase
