import cv2
import os
import sys
import time

# Важный патч DLL-конфликтов
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Backend.traffic_light_pipeline import TrafficLightPipeline

# Цвет рамки bbox по распознанному состоянию
STATE_BOX_COLORS = {
    "red":       (0,   0,   255),
    "green":     (0,   255,  50),
    "yellow":    (0,   215, 255),
    "walk":      (0,   255,  50),
    "dont_walk": (0,   0,   255),
    "unknown":   (180, 180, 180),
}
# Подсветка метки по size_class
SIZE_LABEL_COLORS = {
    "small":  (100, 100, 100),  # серый  - только факт наличия
    "medium": (200, 130,  0),   # оранжевый - пробуем цвет
    "large":  (0,  160,  80),   # зелёный - уверенная классификация
}


def draw_results(frame, results, roi_mode):
    h, w = frame.shape[:2]

    # Линия ROI
    if roi_mode:
        roi_y = int(h * 0.6)
        cv2.line(frame, (0, roi_y), (w, roi_y), (255, 200, 0), 1)
        cv2.putText(frame, "ROI (60%)", (4, roi_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1)

    for r in results:
        bx, by, bw, bh = r["bbox"]
        state      = r["state"]
        size_class = r.get("size_class", "?")
        tl_type    = r["type"]
        phrase     = r["phrase"]

        box_c  = STATE_BOX_COLORS.get(state, (200, 200, 200))
        lbl_bg = SIZE_LABEL_COLORS.get(size_class, (80, 80, 80))

        # Рамка bbox
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_c, 2)

        # Метка: [size] type | state
        label = f"[{size_class}] {tl_type} | {state}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (bx, by - th - 8), (bx + tw + 6, by), lbl_bg, -1)
        cv2.putText(frame, label, (bx + 3, by - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Итоговая фраза под bbox
        if phrase:
            cv2.putText(frame, phrase, (bx, by + bh + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)

    return frame


def test():
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ ПОДСИСТЕМЫ СВЕТОФОРОВ")
    print("  [r] — переключить ROI режим")
    print("  [p] — пауза / продолжение")
    print("  [q] / ESC — выход")
    print("=" * 50)

    roi_mode = True
    pipeline = TrafficLightPipeline(roi_mode=roi_mode)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Не удалось открыть камеру.")
        return

    print("\n[OK] Камера открыта. Наведите на светофор или его фото.\n")

    paused = False
    fps = 0.0
    t_prev = time.time()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            roi_mode = not roi_mode
            pipeline.roi_mode = roi_mode
            for h_obj in pipeline._history.values():
                h_obj.reset()
            print(f"[ROI] {'включён' if roi_mode else 'выключен'}")
        if key == ord('p'):
            paused = not paused
            print("[ПАУЗА]" if paused else "[ПРОДОЛЖЕНИЕ]")

        if paused:
            continue

        ret, frame = cap.read()
        if not ret:
            break

        h_fr, w_fr = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pipeline.process_frame(rgb, h_fr, w_fr)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now

        # Подпись статуса
        roi_lbl = "ROI=ON" if roi_mode else "ROI=OFF"
        cv2.putText(frame, f"FPS:{fps:.1f}  {roi_lbl}  [r][p][q]",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        frame = draw_results(frame, results, roi_mode)
        cv2.imshow("Traffic Light Pipeline Tester", frame)

        # Консольный вывод
        for r in results:
            if r["phrase"]:
                print(f"  >> {r['phrase']}  "
                      f"[size={r.get('size_class','?')} state={r['state']}]")

    cap.release()
    cv2.destroyAllWindows()
    print("Завершено.")


if __name__ == "__main__":
    test()
