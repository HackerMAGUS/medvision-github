"""
test_filtration_logic.py
-------------------------
Тест новой многоуровневой системы фильтрации и стабилизации (без камеры).

Запуск:
  python test_filtration_logic.py
"""

import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Backend.decision_layer import DecisionLayer
from Backend.priority_manager import PriorityManager
from Backend.phrase_builder import PhraseBuilder

# Имитация MediaPipe структур
class _BBox:
    def __init__(self, x, y, w, h):
        self.origin_x, self.origin_y, self.width, self.height = x, y, w, h

class _Cat:
    def __init__(self, name, score):
        self.category_name, self.score = name, score

class _Det:
    def __init__(self, name, x, y, w, h, score=0.85):
        self.bounding_box = _BBox(x, y, w, h)
        self.categories   = [_Cat(name, score)]

class _Result:
    def __init__(self, dets):
        self.detections = dets

W, H = 640, 480

def run_simulation(title, frames_data, safe_mode=True):
    dl = DecisionLayer(near_threshold=0.6, mid_threshold=0.3)
    pm = PriorityManager(safe_mode=safe_mode)
    pb = PhraseBuilder(language="ru")
    
    print(f"\n{'='*70}")
    print(f" SCENARIO: {title}")
    print(f"{'='*70}")
    
    curr_time = 0.0
    for i, dets in enumerate(frames_data):
        curr_time += 0.3 # Имитация ~3 FPS
        print(f"\n--- Frame {i+1} ---")
        
        # Подаем сырые детекции
        res = _Result(dets)
        stable_tracks, zones = dl.process(res, W, H)
        
        print(f" Raw Detections: {len(dets)}")
        print(f" Stable Tracks : {len(stable_tracks)} (те, что прошли temporal filter & confidence)")
        
        for t in stable_tracks:
            blk = "[BLK]" if t.is_blocking_path else ""
            print(f"   Track ID:{t.track_id:2d} | {t.label:10s} | {t.zone:6s} | {t.dist:4s} | {blk}")
            
        should_speak, pri, lvl, best_cand = pm.evaluate_candidates(stable_tracks, curr_time)
        
        if not best_cand:
            print(f" --> SILENT (no valid candidates)")
        else:
            phrase = pb.build_phrase(best_cand.label, best_cand.zone, best_cand.dist)
            if should_speak:
                print(f" --> SAY: '{phrase}' (Level: {lvl}, Priority: {pri})")
            else:
                print(f" --> SILENT: '{phrase}' is muted/on cooldown (Level: {lvl})")
                
# =========================================================================

# СЦЕНАРИЙ 1: Дальние объекты и мелкие шумы игнорируются
frames_1 = [
    # Кадр 1: Только мелкие и дальние объекты, низкий скор
    [
        _Det("bench", 10, 10, 20, 20, score=0.8), # Мелкий, далеко
        _Det("person", 300, 10, 10, 20, score=0.2), # Низкий confidence
    ],
    # Кадр 2: то же самое
    [
        _Det("bench", 10, 10, 20, 20, score=0.8), 
    ],
]
run_simulation("1. Far away & Noise (должны быть полностью отфильтрованы)", frames_1)


# СЦЕНАРИЙ 2: Объекты по бокам не мешают движению
frames_2 = [
    [
        _Det("car", 550, 100, 80, 160), # Справа, средняя дистанция
        _Det("person", 10, 50, 60, 120), # Слева
    ],
    [
        _Det("car", 550, 100, 80, 160), 
        _Det("person", 10, 50, 60, 120),
    ],
    [
        _Det("car", 550, 100, 80, 160), 
        _Det("person", 10, 50, 60, 120),
    ],
]
run_simulation("2. Side objects (stable, but silent in SafeMode unless HIGH)", frames_2, safe_mode=True)


# СЦЕНАРИЙ 3: Близкий объект впереди (obstacle blocking path) + temporal stabilization
frames_3 = [
    # Кадр 1: Появилось препятствие. Пока seen_count = 1, оно нестабильно!
    [
        _Det("wall_blob", 200, 100, 240, 300), # По центру, огромное
    ],
    # Кадр 2: Стабильно
    [
        _Det("wall_blob", 200, 100, 240, 300), 
    ],
    # Кадр 3: Пропуск детектора (моргнуло). Объект должен остаться (lost_count=1)
    [],
    # Кадр 4: Снова детекция
    [
        _Det("wall_blob", 200, 100, 240, 300), 
    ],
]
run_simulation("3. Close Center Obstacle + Blink Resistance", frames_3)


# СЦЕНАРИЙ 4: Несколько объектов, важен только один (выбор приоритета)
frames_4 = [
    [
        _Det("bench", 10, 100, 100, 200), # Слева
        _Det("dog", 500, 50, 80, 100),    # Справа далеко
        _Det("person", 280, 200, 80, 250),# Центр, близко! (Важнее всех)
    ],
    [
        _Det("bench", 10, 100, 100, 200),
        _Det("dog", 500, 50, 80, 100),
        _Det("person", 280, 200, 80, 250), 
    ]
]
run_simulation("4. Multiple objects ( Priority manager sorts them out )", frames_4)

print("\nDONE!")
