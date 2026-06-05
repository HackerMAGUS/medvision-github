"""
Тестировщик Traffic Light Pipeline на видео.

Использование:
  python test_traffic_light_video.py                   # вебкамера (ID 0)
  python test_traffic_light_video.py path/to/video.mp4 # видеофайл
  python test_traffic_light_video.py --no-roi          # без ROI режима

Клавиши во время работы:
  q / ESC  - выйти
  r        - включить/выключить ROI режим
  p        - пауза
"""
import sys
import os
import cv2
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Backend.traffic_light_pipeline import TrafficLightPipeline

# --- Настройки ---
USE_ROI  = "--no-roi" not in sys.argv
SOURCE   = next((a for a in sys.argv[1:] if not a.startswith("--")), 0)
# Если SOURCE - число, передаём int, иначе строку (путь к файлу)
try:
    SOURCE = int(SOURCE)
except (ValueError, TypeError):
    pass

# Цвета для отрисовки по size_class
SIZE_COLORS = {
    "small":  (128, 128, 128),  # серый  - только факт
    "medium": (0,   200, 255),  # голубой - пробуем цвет
    "large":  (0,   255, 128),  # зелёный - уверенная классификация
}
# Цвета рамки по state
STATE_COLORS = {
    "red":       (0,   0,   255),
    "green":     (0,   255,  0),
    "yellow":    (0,   210, 255),
    "walk":      (0,   255,  0),
    "dont_walk": (0,   0,   255),
    "unknown":   (180, 180, 180),
}

def draw_overlay(frame, results, roi_mode, fps):
    h, w = frame.shape[:2]

    # ROI линия
    if roi_mode:
        roi_y = int(h * 0.6)
        cv2.line(frame, (0, roi_y), (w, roi_y), (255, 200, 0), 1)
        cv2.putText(frame, "ROI boundary", (5, roi_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

    for r in results:
        x, bx, bw, bh = r["bbox"][0], r["bbox"][1], r["bbox"][2], r["bbox"][3]
        # Распаковываем правильно
        bx, by, bw, bh = r["bbox"]
        state      = r["state"]
        size_class = r.get("size_class", "medium")
        phrase     = r["phrase"]

        box_color  = STATE_COLORS.get(state, (180, 180, 180))
        label_bg   = SIZE_COLORS.get(size_class, (200, 200, 200))

        # Рамка
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_color, 2)

        # Метка
        label = f"[{size_class}] {r['type']} | {state}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (bx, by - th - 6), (bx + tw + 4, by), label_bg, -1)
        cv2.putText(frame, label, (bx + 2, by - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Фраза
        if phrase:
            cv2.putText(frame, phrase, (bx, by + bh + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    # Статус
    mode_label = "ROI=ON" if roi_mode else "ROI=OFF"
    cv2.putText(frame, f"FPS: {fps:.1f}  {mode_label}  [r]=toggle roi  [p]=pause  [q]=quit",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def main():
    print(f"[Traffic Light Tester] Source: {SOURCE}  ROI: {USE_ROI}")
    pipeline = TrafficLightPipeline(roi_mode=USE_ROI)
    roi_mode = USE_ROI

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть: {SOURCE}")
        return

    paused = False
    fps    = 0.0
    t_prev = time.time()

    print("Открываю окно... Нажмите q/ESC для выхода, r - ROI, p - пауза.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            roi_mode = not roi_mode
            pipeline.roi_mode = roi_mode
            # Сбрасываем историю при смене режима
            for h in pipeline._history.values():
                h.reset()
            print(f"[ROI] {'ON' if roi_mode else 'OFF'}")
        if key == ord('p'):
            paused = not paused
            print("[PAUSED]" if paused else "[RESUMED]")

        if paused:
            continue

        ret, frame = cap.read()
        if not ret:
            # Файл закончился - можно зациклить или выйти
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for h in pipeline._history.values():
                h.reset()
            continue

        h_fr, w_fr = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pipeline.process_frame(rgb, h_fr, w_fr)

        # FPS
        now   = time.time()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        display = draw_overlay(frame.copy(), results, roi_mode, fps)
        cv2.imshow("Traffic Light Pipeline Test", display)

        # Вывод в консоль для зафиксированных результатов
        for r in results:
            if r["phrase"]:
                print(f" >> {r['phrase']}  (state={r['state']}, size={r['size_class']})")

    cap.release()
    cv2.destroyAllWindows()
    print("Завершено.")


if __name__ == "__main__":
    main()
