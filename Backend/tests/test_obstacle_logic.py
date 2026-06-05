"""
Тест obstacle/navigation логики WITHOUT камеры.

Эмулирует разные детекции и проверяет, что DecisionLayer и PhraseBuilder
формируют правильные фразы для сценариев:
  - впереди стена / нераспознанный объект (blocked-path heuristic)
  - впереди человек
  - впереди машина, близко
  - путь свободен
  - боковые объекты

Запуск:
  python test_obstacle_logic.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Backend.decision_layer import DecisionLayer, BLOCK_AREA_NEAR, BLOCK_AREA_MID
from Backend.phrase_builder  import PhraseBuilder

# -----------------------------------------------------------------------
# Фиктивный DetectionResult для тестов без реальной модели
# -----------------------------------------------------------------------
class FakeBBox:
    def __init__(self, x, y, w, h):
        self.origin_x = x
        self.origin_y = y
        self.width    = w
        self.height   = h

class FakeCategory:
    def __init__(self, name, score=0.9):
        self.category_name = name
        self.score         = score

class FakeDetection:
    def __init__(self, category_name, x, y, w, h, score=0.9):
        self.bounding_box = FakeBBox(x, y, w, h)
        self.categories   = [FakeCategory(category_name, score)]

class FakeResult:
    def __init__(self, detections):
        self.detections = detections


# -----------------------------------------------------------------------
# Утилита для вывода
# -----------------------------------------------------------------------
def run(title, detections, frame_w=640, frame_h=480):
    result = FakeResult(detections)
    dl     = DecisionLayer(near_threshold=0.6, mid_threshold=0.3)
    pb     = PhraseBuilder(language="ru")

    label, zone, distance, zones = dl.process(result, frame_w, frame_h)
    phrase = pb.build_phrase(label, zone, distance) if label else "(тишина)"

    ok_mark = "[OK]" if label is not None else "[--]"
    print(f"\n{ok_mark} {title}")
    print(f"    label={label!r:15} zone={zone!r:8} dist={distance!r:6}")
    print(f"    FRAZA: \"{phrase}\"")


# -----------------------------------------------------------------------
# Сценарии
# -----------------------------------------------------------------------
W, H = 640, 480
CENTER_X = int(W * 0.33)  # начало центральной зоны

print("=" * 60)
print(" ТЕСТ: Obstacle / Navigation Logic")
print(f" Пороги blocked-path: NEAR={BLOCK_AREA_NEAR:.0%}  MID={BLOCK_AREA_MID:.0%}")
print("=" * 60)

# 1. Стена/нераспознанный объект занимает ~30% кадра в центре
#    Bbox: от x=160 до x=480 (центр), высота 200px → area = 320*200 = 64000 → ratio ≈ 20.8%
run("Стена в центре (крупный нераспознанный объект)",
    [FakeDetection("wall_unknown_class", x=160, y=100, w=320, h=200)])

# 2. Нераспознанный объект умеренного размера (>4% < 10%) → "препятствие"
#    Bbox 160x120 в центре → area = 19200 → ratio = 6.25%
run("Средний нераспознанный объект в центре",
    [FakeDetection("random_blob", x=200, y=160, w=160, h=120)])

# 3. Маленький нераспознанный объект в центре (< 4%) → тишина
#    Bbox 60x40 → area = 2400 → ratio = 0.78%
run("Malen'kiy ob'ekt v centre (nizhe poroga -> tishina dopustima)",
    [FakeDetection("tiny_thing", x=280, y=220, w=60, h=40)])

# 4. Человек впереди, средняя дистанция
#    rel_h = 150/480 = 0.31 -> mid
run("Chelovek vperedi (srednyaya distanciya)",
    [FakeDetection("person", x=220, y=165, w=100, h=150)])

# 5. Машина, близко (rel_h = 350/480 = 0.73 -> near)
run("Mashina vperedi, blizko",
    [FakeDetection("car", x=150, y=65, w=290, h=350)])

# 6. Человек справа, нет препятствия в центре
run("Chelovek sprava (centr svoboden)",
    [FakeDetection("person", x=500, y=200, w=80, h=150)])

# 7. Пустой кадр -> тишина
run("Pustoy kadr (put' svoboden)",
    [])

# 8. Нераспознанный объект занимает центр И распознанный объект сбоку
#    -> навигационный объект с приоритетом
run("Neraspoznannyy v centre + chelovek sleva",
    [
        FakeDetection("wall_blob", x=200, y=80,  w=200, h=200),  # центр, ~13%
        FakeDetection("person",   x=10,  y=200,  w=80,  h=180),  # слева
    ])

# 9. Собака в центре
run("Sobaka vperedi",
    [FakeDetection("dog", x=240, y=200, w=100, h=160)])

# 10. Незнакомый label (не в NAVIGATION_LABELS, крупный)
#     -> должен сработать blocked-path heuristic
run("Neznakomyy label 'suitcase' (krupnyy, v centre)",
    [FakeDetection("suitcase", x=180, y=80, w=250, h=280)])

print("\n" + "=" * 60)
print("Gotovo.")
print("Scenarii 1,2,4,5,8,9,10 - dolzhna byt' fraza [OK]")
print("Scenariy 3 (malen'kiy) - [--] tishina - eto normal'no")
print("Scenariy 6 (sprava) - OK, chelovek sprava")
print("Scenariy 7 (pustyy) - [--] tishina - eto normal'no")
print("=" * 60)
