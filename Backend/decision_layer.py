import math

# Объекты, название которых важно озвучить явно.
NAVIGATION_LABELS = {
    "person", "car", "bicycle", "motorcycle", "bus",
    "truck", "train", "dog", "chair", "dining table",
    "bench", "potted plant", "stop sign", "pole", "obstacle",
    "barrier", "stairs_down"
}

# Минимальный confidence score для фильтрации "миражей"
BLOCK_MIN_SCORE = 0.35

class TrackedObject:
    """Олицетворяет физический объект, отслеживаемый во времени."""
    def __init__(self, track_id, label, zone, dist, cx, cy, area_r, bbox=None):
        self.track_id = track_id
        self.label = label
        self.zone = zone
        self.dist = dist
        self.cx = cx
        self.cy = cy
        self.area_r = area_r
        self.bbox = bbox
        
        self.seen_count = 1
        self.lost_count = 0
        self.is_stable = False
        self.is_blocking_path = False
        self._eval_blocking()
        
    def update(self, zone, dist, cx, cy, area_r, bbox=None):
        self.zone = zone
        self.dist = dist
        self.cx = cx
        self.cy = cy
        self.area_r = area_r
        if bbox is not None:
            self.bbox = bbox
        
        self.seen_count += 1
        self.lost_count = 0
        if self.seen_count >= 2: # Объект подтвердился во времени
            self.is_stable = True
            
        self._eval_blocking()
            
    def _eval_blocking(self):
        """Path relevance logic: мешает ли объект движению."""
        self.is_blocking_path = False
        
        # Близко и по центру -> почти всегда перекрывает путь
        if self.dist == "near" and self.zone == "center":
            self.is_blocking_path = True
            
        # Занимает весомую часть экрана в центре (машина на пару метров или человек)
        elif self.zone == "center" and self.area_r >= 0.03:
            self.is_blocking_path = True
            
        # Крупный объект сбоку или очень близко (можем задеть плечом/палкой)
        elif self.dist == "near" and self.area_r >= 0.15:
            self.is_blocking_path = True


class DecisionLayer:
    def __init__(self, near_threshold=0.6, mid_threshold=0.3):
        self.near = near_threshold
        self.mid  = mid_threshold
        
        self.tracks = []
        self._next_id = 1
        
    def _zone(self, rel_cx: float) -> str:
        if rel_cx < 0.33: return "left"
        if rel_cx > 0.66: return "right"
        return "center"

    def _distance(self, bh: float, frame_h: float) -> str:
        rel_h = bh / frame_h
        if rel_h > self.near: return "near"
        if rel_h > self.mid:  return "mid"
        return "far"

    def process(self, detection_result, frame_width, frame_height):
        """
        Candidate Filtering Layer + Temporal Stabilizer.
        Возвращает:
           stable_tracks: список подтвержденных TrackedObject
           zones_occupied: dict (состояние зон для UI)
        """
        zones_occupied = {"left": False, "center": False, "right": False}
        
        # 1. Извлекаем и фильтруем текущие детекции (Candidates)
        current_candidates = []
        if detection_result and hasattr(detection_result, 'detections') and detection_result.detections:
            frame_area = frame_width * frame_height
            for d in detection_result.detections:
                cat = d.categories[0]
                if cat.score < BLOCK_MIN_SCORE: 
                    continue # Игнорируем неуверенные детекции
                
                bbox = d.bounding_box
                cx = bbox.origin_x + bbox.width / 2
                cy = bbox.origin_y + bbox.height / 2
                zone = self._zone(cx / frame_width)
                dist = self._distance(bbox.height, frame_height)
                area_r = (bbox.width * bbox.height) / frame_area
                label = cat.category_name
                bbox_tuple = (
                    int(bbox.origin_x),
                    int(bbox.origin_y),
                    int(bbox.width),
                    int(bbox.height),
                )
                
                # Фильтрация мусора и мелких неизвестных объектов
                if label not in NAVIGATION_LABELS:
                    if zone == "center" and area_r >= 0.04:
                        label = "obstacle" # Крупный неизвестный в центре -> препятствие
                    else:
                        continue # Мелкий или боковой мусор отбрасываем
                
                # Совсем мелкие объекты (меньше 1% кадра) убираем как мусор/ошибку
                if area_r < 0.01:
                    continue
                
                zones_occupied[zone] = True
                current_candidates.append({
                    "label": label, "zone": zone, "dist": dist,
                    "cx": cx, "cy": cy, "area_r": area_r,
                    "bbox": bbox_tuple
                })
                
        # 2. Матчинг к существующим трекам (Temporal Stabilization)
        matched_tracks = set()
        max_dist = frame_width * 0.20 # 20% кадра - макс смещение объекта за 1 тик
        
        for cand in current_candidates:
            best_trk = None
            best_d = max_dist
            
            for trk in self.tracks:
                if trk in matched_tracks: 
                    continue
                if trk.label == cand["label"]:
                    d = math.hypot(trk.cx - cand["cx"], trk.cy - cand["cy"])
                    if d < best_d:
                        best_d = d
                        best_trk = trk
                        
            if best_trk:
                best_trk.update(cand["zone"], cand["dist"], cand["cx"], cand["cy"], cand["area_r"], cand["bbox"])
                matched_tracks.add(best_trk)
            else:
                new_trk = TrackedObject(
                    self._next_id,
                    cand["label"],
                    cand["zone"],
                    cand["dist"],
                    cand["cx"],
                    cand["cy"],
                    cand["area_r"],
                    cand["bbox"],
                )
                self._next_id += 1
                self.tracks.append(new_trk)
                matched_tracks.add(new_trk)
                
        # 3. Обновляем lost_count и удаляем исчезнувшие объекты (Graceful loss)
        for trk in self.tracks[:]:
            if trk not in matched_tracks:
                trk.lost_count += 1
                if trk.lost_count >= 3: # Считаем объект ушедшим только после 3 пропусков
                    self.tracks.remove(trk)
                    
        # 4. Выборка только стабильных объектов для Priority Manager
        stable_tracks = [t for t in self.tracks if t.is_stable]
        return stable_tracks, zones_occupied
