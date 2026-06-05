LABELS = {
    "person": "odam",
    "car": "mashina",
    "bicycle": "velosiped",
    "motorcycle": "mototsikl",
    "bus": "avtobus",
    "train": "poyezd",
    "truck": "yuk mashinasi",
    "traffic light": "svetofor",
    "stop sign": "to'xtash belgisi",
    "bench": "o'rindiq",
    "chair": "stul",
    "potted plant": "gul tuvagi",
    "bed": "karavot",
    "dining table": "stol",
    "tv": "televizor",
    "door": "eshik",
    "pole": "ustun",
    "obstacle": "to'siq",
    "unknown": "to'siq",
}

ZONES = {
    "left": "Chapda",
    "center": "Oldinda",
    "right": "O'ngda",
    "unknown": "",
}

DISTANCES = {
    "near": "yaqin",
    "mid": "",
    "far": "",
    "unknown": "",
}


def format_phrase(zone_text, label_text, distance_text):
    phrase = f"{zone_text} {label_text}".strip()
    if distance_text:
        phrase += f", {distance_text}"
    return phrase
