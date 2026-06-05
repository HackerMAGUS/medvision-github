"""
test_priority_logic.py
-----------------------
Тест priority manager без камеры — 3 ключевых сценария + дополнительные.

Запуск:
  python test_priority_logic.py
"""
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Backend.priority_manager import PriorityManager, compute_priority_score, score_to_level
from Backend.decision_layer   import DecisionLayer
from Backend.phrase_builder   import PhraseBuilder

# -----------------------------------------------------------------------
# Fake detection helpers (без реальной модели)
# -----------------------------------------------------------------------
class _BBox:
    def __init__(self, x, y, w, h):
        self.origin_x, self.origin_y, self.width, self.height = x, y, w, h

class _Cat:
    def __init__(self, name, score=0.85):
        self.category_name, self.score = name, score

class _Det:
    def __init__(self, name, x, y, w, h, score=0.85):
        self.bounding_box = _BBox(x, y, w, h)
        self.categories   = [_Cat(name, score)]

class _Result:
    def __init__(self, dets):
        self.detections = dets

W, H = 640, 480
FRAME_AREA = W * H

# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------
def run_scenario(title, dets, safe_mode=True, cycles=3):
    """
    Прогоняет сценарий через N циклов (имитация video frames).
    Выводит priority score каждого детектируемого объекта и финальное решение.
    """
    dl  = DecisionLayer(near_threshold=0.6, mid_threshold=0.3)
    pm  = PriorityManager(safe_mode=safe_mode)
    pb  = PhraseBuilder(language="ru")
    res = _Result(dets)

    print(f"\n{'='*55}")
    print(f"  SCENARIO: {title}")
    print(f"  safe_mode={safe_mode}, cycles={cycles}")
    print(f"{'='*55}")

    # Показать scores всех объектов в кадре
    print("  [Detections priority scores]:")
    for d in dets:
        lbl   = d.categories[0].category_name
        bh    = d.bounding_box.height
        cx    = d.bounding_box.origin_x + d.bounding_box.width/2
        zone  = "center" if W*0.33 < cx < W*0.66 else ("left" if cx < W*0.33 else "right")
        dist  = "near" if bh/H > 0.6 else ("mid" if bh/H > 0.3 else "far")
        sc    = compute_priority_score(lbl, zone, dist)
        lvl   = score_to_level(sc)
        print(f"    {lbl:20s} zone={zone:6s} dist={dist:4s}  score={sc:3d}  level={lvl}")

    label, zone, distance, zones = dl.process(res, W, H)
    print(f"\n  [DecisionLayer] best -> label={label!r} zone={zone!r} dist={distance!r}")

    spoken = []
    t = time.time()
    for i in range(cycles):
        t += 0.4   # имитируем ~2.5 fps
        should, pri, lvl = pm.evaluate(label, zone, distance, t)
        if should:
            phrase = pb.build_phrase(label, zone, distance) if label else ""
            spoken.append(f"  cycle {i+1}: SAY (pri={pri}, lvl={lvl}) -> \"{phrase}\"")
        else:
            spoken.append(f"  cycle {i+1}: SILENT (lvl={lvl})")

    for s in spoken:
        print(s)

    if not any("SAY" in s for s in spoken):
        if label is None:
            print("  -> Korrektno: put' svoboden, molchim.")
        else:
            print("  -> VNIMANIE: ob'ekt est', no ne ozvochen!")
    print()


# -----------------------------------------------------------------------
# SCENARIO 1: Близкое препятствие впереди + много объектов по бокам
# -----------------------------------------------------------------------
run_scenario(
    "SCENARIO 1: Close obstacle ahead + busy sides",
    dets=[
        # Центр — большой нераспознанный объект (стена/дверь) 30% кадра
        _Det("unknown_blocker", x=160, y=50,  w=320, h=300),
        # Боковые — люди, велосипеды
        _Det("person",    x=10,  y=150, w=80,  h=180),
        _Det("bicycle",   x=550, y=200, w=60,  h=160),
        _Det("person",    x=580, y=100, w=70,  h=200),
        _Det("bench",     x=520, y=300, w=90,  h=80),
        _Det("potted plant", x=20, y=350, w=50, h=70),
    ],
    safe_mode=True,
    cycles=3,
)

# -----------------------------------------------------------------------
# SCENARIO 2: Только боковые объекты, центр свободен
# -----------------------------------------------------------------------
run_scenario(
    "SCENARIO 2: Only side objects, center clear",
    dets=[
        _Det("person",   x=10,  y=150, w=80,  h=200),
        _Det("bicycle",  x=550, y=150, w=70,  h=180),
        _Det("bench",    x=0,   y=320, w=100, h=80),
        _Det("car",      x=530, y=100, w=100, h=120),
    ],
    safe_mode=True,
    cycles=3,
)

# -----------------------------------------------------------------------
# SCENARIO 3: Несколько объектов, центр свободен, один важный справа
# -----------------------------------------------------------------------
run_scenario(
    "SCENARIO 3: Several objects, center free, car on right",
    dets=[
        _Det("car",         x=480, y=80,  w=140, h=200),   # справа, большая машина
        _Det("person",      x=500, y=200, w=60,  h=150),   # справа, человек
        _Det("bench",       x=10,  y=320, w=100, h=70),    # слева, скамейка
        _Det("potted plant",x=490, y=350, w=50,  h=60),    # справа, мелочь
    ],
    safe_mode=True,
    cycles=4,
)

# -----------------------------------------------------------------------
# EXTRA: Человек в центре, близко (HIGH) -> говорим с cycle 1
# -----------------------------------------------------------------------
run_scenario(
    "EXTRA: Person center near -> HIGH, immediate speak",
    dets=[
        _Det("person", x=230, y=0, w=160, h=380),  # центр, rel_h=0.79 -> near
    ],
    safe_mode=True,
    cycles=3,
)

# -----------------------------------------------------------------------
# EXTRA: Safe Mode OFF — говорим даже о скамейке
# -----------------------------------------------------------------------
run_scenario(
    "EXTRA: Safe Mode OFF -> bench on side also spoken",
    dets=[
        _Det("bench",  x=10,  y=300, w=120, h=100),
        _Det("person", x=300, y=200, w=80,  h=140),
    ],
    safe_mode=False,
    cycles=4,
)

print("="*55)
print("Test complete.")
print("Expected:")
print("  Scenario 1 -> SAY 'Vperedi prepyatstvie, blizko'")
print("  Scenario 2 -> SILENT (center clear)")
print("  Scenario 3 -> maybe SAY about car on right after 2 cycles")
print("  EXTRA 1    -> SAY on cycle 1 (HIGH, no stabilisation wait)")
print("  EXTRA 2    -> SAY bench/person even from side (safe_mode=OFF)")
print("="*55)
