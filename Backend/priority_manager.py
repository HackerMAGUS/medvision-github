import time

_TYPE_HIGH = {"obstacle", "pole", "stop sign", "barrier", "stairs_down"}
_TYPE_MEDIUM = {"person", "car", "bicycle", "motorcycle", "bus", "truck", "train", "dog"}
_TYPE_LOW = {"bench", "chair", "potted plant", "dining table", "bed", "tv", "bird", "cat"}

_SAFE_MODE_MEDIUM_LABELS = {"person", "car", "bicycle", "motorcycle", "bus", "truck", "dog"}

# Speech budget (cooldown в секундах)
_COOLDOWN = {
    "HIGH":   3.0,
    "MEDIUM": 6.0,
    "LOW":    10.0,
}

def compute_priority_score(obj) -> int:
    """Единая оценка кандидата. Возвращает int score."""
    score = 0
    
    # 1. Base Class Importance
    if obj.label in _TYPE_HIGH: score += 50
    elif obj.label in _TYPE_MEDIUM: score += 20
    else: score += 5
    
    # 2. Blocking path relevance (самое главное для помощника)
    # Если крупный объект мешает на пути - огромный бонус
    if obj.is_blocking_path:
        score += 100 
        
    # 3. Zone Importance
    if obj.zone == "center": score += 30
    
    # 4. Distance Importance
    if obj.dist == "near": score += 50
    elif obj.dist == "mid": score += 15
    
    # 5. Size 
    score += int(obj.area_r * 100)
    
    # 6. Temporal Stability Bonus (чем дольше видим, тем увереннее)
    score += min(20, obj.seen_count * 2) 
    
    return score

def score_to_level(score: int) -> str:
    """Определение уровня угрозы."""
    if score >= 120: return "HIGH"
    if score >= 60:  return "MEDIUM"
    return "LOW"

class PriorityManager:
    """
    Принимает список подтвержденных TrackedObject от DecisionLayer,
    выбирает самого приоритетного и управляет речь-повторениями.
    """
    def __init__(self, safe_mode=True):
        self.safe_mode = safe_mode
        self.last_spoken_id = None
        self.last_spoken_time = 0.0
        self.last_spoken_dist = None
        self.last_spoken_label = None
        self.last_spoken_zone = None
        self.last_spoken_blocking = False
        
    def evaluate_candidates(self, candidates, current_time=None):
        """
        Возвращает:
           (should_speak, speech_priority, level, best_cand)
        """
        if current_time is None: 
            current_time = time.time()
        
        best_cand = None
        best_score = -1
        
        # Выбираем ТОП-1 кандидата
        for cand in candidates:
            # --- ГЛОБАЛЬНЫЕ ПРАВИЛА ФИЛЬТРАЦИИ ---
            
            # 1. Слишком далеко - игнорируем, они не представляют опасности
            if cand.dist == "far":
                continue
                
            # 2. Сбоку и не мешает движению - игнорируем
            if cand.zone in ("left", "right") and not cand.is_blocking_path:
                continue
                
            score = compute_priority_score(cand)
            lvl = score_to_level(score)
            
            # Safe Mode Rules
            if self.safe_mode:
                if lvl == "LOW": continue
                if lvl == "MEDIUM" and cand.label not in _SAFE_MODE_MEDIUM_LABELS: continue
                
            if score > best_score:
                best_score = score
                best_cand = cand
                
        # Если кандидатов после фильтрации нет
        if not best_cand:
            return False, 0, "NONE", None
            
        level = score_to_level(best_score)
        
        # --- Speech Repeat Logic ---
        is_same_object = (best_cand.track_id == self.last_spoken_id)
        elapsed = current_time - self.last_spoken_time
        
        # Флаги важности события
        became_closer = is_same_object and (
            (self.last_spoken_dist != "near" and best_cand.dist == "near")
        )
        
        became_blocking = is_same_object and (
            not self.last_spoken_blocking and best_cand.is_blocking_path
        )
        
        is_new_threat = (not is_same_object) and (level == "HIGH" or best_score >= 120)
        
        cooldown = _COOLDOWN.get(level, 5.0)
        should_speak = False
        
        # 1. Стал ближе или начал перекрывать путь - предупреждаем сразу
        if became_closer or became_blocking:
            should_speak = True
        # 2. Новая серьезная угроза - может перебить через 1 сек
        elif is_new_threat and elapsed >= 1.0:
            should_speak = True
        # 3. Напоминание об объекте (cooldown timer)
        elif elapsed >= cooldown:
            should_speak = True
            
        if should_speak:
            self.last_spoken_id = best_cand.track_id
            self.last_spoken_time = current_time
            self.last_spoken_dist = best_cand.dist
            self.last_spoken_label = best_cand.label
            self.last_spoken_zone = best_cand.zone
            self.last_spoken_blocking = best_cand.is_blocking_path
            
            pri = 1 if level == "HIGH" else 0
            return True, pri, level, best_cand
            
        return False, 0, level, best_cand
