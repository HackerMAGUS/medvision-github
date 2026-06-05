from collections import deque
from Backend.traffic_light_detector import TrafficLightDetector
from Backend.traffic_light_classifier import TrafficLightClassifier
from Backend.traffic_light_formatter import TrafficLightFormatter

# --- Пороги размера Bbox (в пикселях) ---
# small:  w < MEDIUM_W или h < MEDIUM_H  -> только факт наличия, цвет не определяем
# medium: выше small, ниже large          -> пробуем определить цвет (HSV)
# large:  выше large                      -> уверенная классификация цвета
SMALL_W,  SMALL_H  = 20, 30
MEDIUM_W, MEDIUM_H = 40, 60

# Temporal Confirmation:
# Светофор нужно заметить в N кадрах подряд, прежде чем озвучить
CONFIRM_FRAMES = 2      # сколько кадров подряд нужно для "светофор есть"
COLOR_CONFIRM_FRAMES = 2 # сколько кадров подряд нужен одинаковый цвет


class _ZoneHistory:
    """Накопитель истории для одной зоны (left/center/right)."""
    def __init__(self):
        self.det_streak = 0          # сколько кадров подряд есть детекция
        self.color_buffer = deque(maxlen=COLOR_CONFIRM_FRAMES)  # история цветов
        self.confirmed = False       # светофор подтверждён
        self.confirmed_color = None  # подтверждённый цвет

    def update(self, state: str) -> tuple[bool, str | None]:
        """
        Обновляет историю.
        Returns: (светофор_подтверждён, цвет_или_None)
        """
        self.det_streak += 1
        self.color_buffer.append(state)

        if self.det_streak >= CONFIRM_FRAMES:
            self.confirmed = True

        # Подтверждаем цвет, только если в буфере все значения одинаковы и не unknown
        confirmed_color = None
        if len(self.color_buffer) == COLOR_CONFIRM_FRAMES:
            if len(set(self.color_buffer)) == 1 and self.color_buffer[0] != "unknown":
                confirmed_color = self.color_buffer[0]
        self.confirmed_color = confirmed_color
        return self.confirmed, self.confirmed_color

    def reset(self):
        self.det_streak = 0
        self.color_buffer.clear()
        self.confirmed = False
        self.confirmed_color = None


class TrafficLightPipeline:
    def __init__(self, decision_layer=None, roi_mode=True):
        self.detector = TrafficLightDetector()
        self.classifier = TrafficLightClassifier()
        self.formatter = TrafficLightFormatter()
        self.decision_layer = decision_layer
        self.roi_mode = roi_mode

        # История по зонам: сбрасывается, если в этой зоне нет детекции
        self._history: dict[str, _ZoneHistory] = {
            "left": _ZoneHistory(),
            "center": _ZoneHistory(),
            "right": _ZoneHistory(),
        }
        # Отслеживаем, какие зоны были активны в текущем кадре
        self._active_zones: set[str] = set()

    def _get_bbox_size_class(self, w: int, h: int) -> str:
        """Возвращает 'small', 'medium' или 'large' по размеру bbox."""
        if w < SMALL_W or h < SMALL_H:
            return "small"
        elif w < MEDIUM_W or h < MEDIUM_H:
            return "medium"
        return "large"

    def process_frame(self, rgb_frame, original_h, original_w):
        """
        Главный оркестратор двухэтапного анализа:
        1. Ищем светофоры (Detection) — с опциональным ROI
        2. Определяем размер bbox (Small/Medium/Large)
        3. Запускаем Classification только для medium/large
        4. Обновляем Temporal Confirmation по зонам
        5. Переводим в текст (Format)
        """
        # 1. Detection (с ROI если нужно)
        detections = self.detector.detect(rgb_frame, roi_mode=self.roi_mode)

        results = []
        self._active_zones.clear()

        for det in detections:
            x, y, w, h = det["bbox"]
            tl_type = det["type"]

            # Базовая фильтрация мусора
            if w < 8 or h < 8:
                continue

            # 2. Определяем размер bbox
            size_class = self._get_bbox_size_class(w, h)

            # 3. Classification
            if size_class == "small":
                # Слишком далеко — цвет не определяем
                state = "unknown"
            else:
                # medium / large — пробуем HSV
                crop = rgb_frame[max(0, y):min(original_h, y+h),
                                  max(0, x):min(original_w, x+w)]
                state = self.classifier.classify(crop, tl_type)

            # 4. Spatial Analysis
            cx = x + w / 2
            area = w * h
            area_ratio = area / (original_h * original_w)

            zone = "center"
            if cx < original_w * 0.33:
                zone = "left"
            elif cx > original_w * 0.66:
                zone = "right"

            distance = "far"
            if area_ratio > 0.15:
                distance = "near"
            elif area_ratio > 0.05:
                distance = "mid"

            # 4a. Temporal Confirmation
            self._active_zones.add(zone)
            confirmed, confirmed_color = self._history[zone].update(state)

            if not confirmed:
                # Ещё не подтверждено — пропускаем, не спамим
                continue

            # Если в истории нет стабильного цвета — остаётся unknown
            final_state = confirmed_color if confirmed_color else "unknown"

            # 5. Formatting
            phrase = self.formatter.format_phrase(tl_type, final_state, zone, distance, size_class)

            results.append({
                "bbox": (x, y, w, h),
                "type": tl_type,
                "state": final_state,
                "size_class": size_class,
                "phrase": phrase
            })

        # Сбрасываем историю для зон, где в этом кадре ничего не было
        for zone, hist in self._history.items():
            if zone not in self._active_zones:
                hist.reset()

        return results
