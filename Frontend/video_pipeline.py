import os
import time

import cv2

from Backend.priority_manager import PriorityManager, compute_priority_score, score_to_level
from Backend.traffic_light_pipeline import TrafficLightPipeline

_TL_EVERY = 2

_TL_STATE_COLORS = {
    "red": (0, 0, 220),
    "green": (0, 210, 60),
    "yellow": (0, 200, 255),
    "walk": (0, 210, 60),
    "dont_walk": (0, 0, 220),
    "unknown": (160, 160, 160),
}

_LEVEL_BOX_COLORS = {
    "HIGH": (0, 0, 255),
    "MEDIUM": (0, 165, 255),
    "LOW": (0, 180, 60),
}


def camera_backend():
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def probe_cameras(max_index=5):
    cameras = []
    backend = camera_backend()
    for index in range(max_index):
        cap = cv2.VideoCapture(index, backend)
        opened = cap.isOpened()
        ret, frame = cap.read() if opened else (False, None)
        if ret and frame is not None:
            h, w = frame.shape[:2]
            kind = "kamera noutbuka" if index == 0 else "tashqi webcam"
            cameras.append(
                {
                    "id": index,
                    "label": f"Kamera {index} - {kind} ({w}x{h})",
                }
            )
        cap.release()
    return cameras


class VideoPipeline:
    def __init__(self, detector, decision_layer, phrase_builder, speech_engine, camera_id=0, safe_mode=True):
        self.detector = detector
        self.decision_layer = decision_layer
        self.phrase_builder = phrase_builder
        self.speech_engine = speech_engine
        self.camera_id = camera_id

        self.cap = cv2.VideoCapture(camera_id, camera_backend())
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")

        self.ai_target_fps = 3
        self.ai_interval = 1.0 / self.ai_target_fps
        self.last_ai_time = 0.0

        self.last_detections = None
        self.current_phrase = "Qidirilmoqda..."
        self._debug_info = "PRIORITY: --"
        self._stable_tracks = []
        self._last_level = "--"
        self._last_frame_size = (0, 0)

        self.priority_mgr = PriorityManager(safe_mode=safe_mode)
        self.tl_pipeline = TrafficLightPipeline(roi_mode=True)
        self._tl_tick = 0
        self._tl_results = []
        self._tl_phrase_cooldown = 5.0
        self._tl_last_spoken = {}

    def _process_traffic_lights(self, rgb_frame, h, w, current_time):
        self._tl_tick += 1
        if self._tl_tick % _TL_EVERY != 0:
            return

        self._tl_results = self.tl_pipeline.process_frame(rgb_frame, h, w)
        for result in self._tl_results:
            phrase = result.get("phrase", "")
            if not phrase:
                continue
            last_t = self._tl_last_spoken.get(phrase, 0.0)
            if current_time - last_t < self._tl_phrase_cooldown:
                continue
            self._tl_last_spoken[phrase] = current_time
            self.speech_engine.say(phrase, priority=0)

    def read_processed_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        h, w = frame.shape[:2]
        self._last_frame_size = (w, h)
        current_time = time.time()

        if (current_time - self.last_ai_time) >= self.ai_interval:
            self.last_ai_time = current_time
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detection_result = self.detector.detect(rgb_frame)
            self.last_detections = detection_result
            stable_tracks, _zones_occupied = self.decision_layer.process(detection_result, w, h)
            self._stable_tracks = stable_tracks

            should_speak, speech_pri, level, best_cand = self.priority_mgr.evaluate_candidates(
                stable_tracks, current_time
            )
            self._last_level = level

            if not best_cand:
                self.current_phrase = "Yo'l ochiq"
                self._debug_info = "PRIORITY: --"
            else:
                phrase = self.phrase_builder.build_phrase(best_cand.label, best_cand.zone, best_cand.dist)
                self.current_phrase = phrase
                block_marker = " [BLK]" if best_cand.is_blocking_path else ""
                self._debug_info = (
                    f"PRI={level} [{best_cand.label}|{best_cand.zone}|{best_cand.dist}] "
                    f"ID:{best_cand.track_id}{block_marker}"
                )
                if should_speak:
                    self.speech_engine.say(phrase, priority=speech_pri)

            self._process_traffic_lights(rgb_frame, h, w, current_time)

        self._draw_overlay(frame, self._stable_tracks, self.current_phrase, self._debug_info, self._tl_results, w, h)
        return frame

    def get_status(self):
        return {
            "phrase": self.current_phrase,
            "debug": self._debug_info,
            "level": self._last_level,
            "objects": len(self._stable_tracks),
            "camera": self.camera_id,
            "size": self._last_frame_size,
        }

    def release(self):
        if self.cap:
            self.cap.release()

    def start(self):
        print("[VideoPipeline] Start. Press 'q' to exit.")
        while True:
            frame = self.read_processed_frame()
            if frame is None:
                break
            cv2.imshow("Real-Time Vision AI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.release()
        cv2.destroyAllWindows()

    def _draw_overlay(self, frame, stable_tracks, text_to_show, debug_info, tl_results, w, h):
        cv2.line(frame, (int(w * 0.33), 0), (int(w * 0.33), h), (235, 235, 235), 1)
        cv2.line(frame, (int(w * 0.66), 0), (int(w * 0.66), h), (235, 235, 235), 1)

        cv2.putText(frame, text_to_show, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(frame, debug_info, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        for trk in stable_tracks:
            sc = compute_priority_score(trk)
            lvl = score_to_level(sc)
            color = _LEVEL_BOX_COLORS.get(lvl, (0, 180, 60))
            label = f"{trk.label} id:{trk.track_id} {'[BLK]' if trk.is_blocking_path else ''}".strip()

            if getattr(trk, "bbox", None):
                bx, by, bw, bh = trk.bbox
                x1 = max(0, min(w - 1, int(bx)))
                y1 = max(0, min(h - 1, int(by)))
                x2 = max(0, min(w - 1, int(bx + bw)))
                y2 = max(0, min(h - 1, int(by + bh)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            else:
                ctx = int(trk.cx)
                cty = int(trk.cy)
                cv2.drawMarker(frame, (ctx, cty), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cv2.putText(frame, label, (ctx - 10, cty - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        for result in tl_results:
            bx, by, bw, bh = result["bbox"]
            state = result["state"]
            size_class = result.get("size_class", "?")
            phrase = result.get("phrase", "")

            color = _TL_STATE_COLORS.get(state, (160, 160, 160))
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(frame, f"TL[{size_class}] {state}", (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            if phrase:
                cv2.putText(frame, phrase, (bx, by + bh + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 140, 220), 1)
