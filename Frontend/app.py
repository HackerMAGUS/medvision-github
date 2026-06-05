import queue
import threading
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from Backend.decision_layer import DecisionLayer
from Backend.detector import Detector
from Backend.phrase_builder import PhraseBuilder
from Backend.speech_engine import SpeechEngine
from Frontend.video_pipeline import VideoPipeline, probe_cameras


class BlindAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MedVision")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)
        self.root.configure(bg="#ffffff")

        self.detector = None
        self.phrase_builder = None
        self.speech_engine = None
        self.pipeline = None
        self.running = False
        self.photo = None
        self.cameras = []
        self.events = queue.Queue()

        self.camera_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Tizim yuklanmoqda...")
        self.phrase_var = tk.StringVar(value="Kutish rejimi")
        self.camera_status_var = tk.StringVar(value="Kamera tanlanmagan")
        self.objects_var = tk.StringVar(value="0")
        self.level_var = tk.StringVar(value="--")

        self._setup_styles()
        self._build_ui()
        self._bind_keys()
        self._load_backend_async()
        self._scan_cameras_async()
        self._poll_events()

    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TCombobox", font=("Segoe UI", 13), padding=8)

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#ffffff")
        header.pack(fill="x", padx=28, pady=(22, 10))

        tk.Label(
            header,
            text="MedVision",
            bg="#ffffff",
            fg="#101828",
            font=("Segoe UI", 26, "bold"),
        ).pack(side="left")

        self.status_badge = tk.Label(
            header,
            textvariable=self.status_var,
            bg="#eef8f1",
            fg="#166534",
            font=("Segoe UI", 12, "bold"),
            padx=18,
            pady=8,
        )
        self.status_badge.pack(side="right")

        body = tk.Frame(self.root, bg="#ffffff")
        body.pack(fill="both", expand=True, padx=28, pady=12)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, minsize=320)
        body.rowconfigure(0, weight=1)

        video_panel = tk.Frame(body, bg="#f8fafc", highlightbackground="#e5e7eb", highlightthickness=1)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        video_panel.rowconfigure(0, weight=1)
        video_panel.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(video_panel, bg="#ffffff", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        self.canvas.bind("<Configure>", lambda _event: self._draw_idle_canvas())

        side = tk.Frame(body, bg="#ffffff")
        side.grid(row=0, column=1, sticky="nsew")

        self._section_label(side, "Kamera")
        self.camera_combo = ttk.Combobox(side, textvariable=self.camera_var, state="readonly")
        self.camera_combo.pack(fill="x", pady=(0, 10))

        self.scan_button = self._button(side, "Kameralarni yangilash", self._scan_cameras_async, "#ffffff", "#111827", "#d0d5dd")
        self.start_button = self._button(side, "Boshlash", self.start, "#111827", "#ffffff", "#111827")
        self.stop_button = self._button(side, "To'xtatish", self.stop, "#ffffff", "#b42318", "#fecdca")

        self._section_label(side, "Holat")
        self._metric(side, "Oxirgi xabar", self.phrase_var, large=True)
        self._metric(side, "Xavf darajasi", self.level_var)
        self._metric(side, "Topilgan obyektlar", self.objects_var)
        self._metric(side, "Faol kamera", self.camera_status_var)

        footer = tk.Frame(self.root, bg="#ffffff")
        footer.pack(fill="x", padx=28, pady=(0, 18))
        tk.Label(
            footer,
            text="Offline Uzbek TTS: Meta MMS-TTS",
            bg="#ffffff",
            fg="#667085",
            font=("Segoe UI", 10),
        ).pack(side="left")

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, bg="#ffffff", fg="#344054", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(8, 8))

    def _button(self, parent, text, command, bg, fg, border):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            highlightbackground=border,
            highlightthickness=1,
            bd=0,
            font=("Segoe UI", 15, "bold"),
            padx=18,
            pady=14,
            cursor="hand2",
        )
        btn.pack(fill="x", pady=7)
        return btn

    def _metric(self, parent, title, variable, large=False):
        frame = tk.Frame(parent, bg="#f9fafb", highlightbackground="#eaecf0", highlightthickness=1)
        frame.pack(fill="x", pady=7)
        tk.Label(frame, text=title, bg="#f9fafb", fg="#667085", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(10, 2))
        tk.Label(
            frame,
            textvariable=variable,
            bg="#f9fafb",
            fg="#101828",
            font=("Segoe UI", 17 if large else 14, "bold"),
            wraplength=270,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 12))

    def _bind_keys(self):
        self.root.bind("<space>", lambda _event: self.start() if not self.running else self.stop())
        self.root.bind("<Escape>", lambda _event: self.stop())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _load_backend_async(self):
        def worker():
            try:
                detector = Detector()
                phrase_builder = PhraseBuilder(language="uz")
                speech_engine = SpeechEngine(cooldown=2.0, ttl=8.0, language="uz")
                self.events.put(("backend_ready", detector, phrase_builder, speech_engine))
            except Exception as exc:
                self.events.put(("error", f"Backend xatosi: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _scan_cameras_async(self):
        self.status_var.set("Kameralar qidirilmoqda...")

        def worker():
            try:
                self.events.put(("cameras", probe_cameras(max_index=6)))
            except Exception as exc:
                self.events.put(("error", f"Kamera qidirishda xato: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_events(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break

            kind = event[0]
            if kind == "backend_ready":
                self.detector, self.phrase_builder, self.speech_engine = event[1:]
                self.status_var.set("Tizim tayyor")
            elif kind == "cameras":
                self.cameras = event[1]
                labels = [cam["label"] for cam in self.cameras]
                self.camera_combo["values"] = labels
                if labels and not self.camera_var.get():
                    self.camera_var.set(labels[0])
                if labels:
                    self.status_var.set("Kamera tayyor")
                else:
                    self.status_var.set("Kamera topilmadi")
            elif kind == "error":
                self.status_var.set(event[1])

        self.root.after(100, self._poll_events)

    def _selected_camera_id(self):
        selected = self.camera_var.get()
        for cam in self.cameras:
            if cam["label"] == selected:
                return cam["id"]
        return 0

    def start(self):
        if self.running:
            return
        if not self.detector or not self.phrase_builder or not self.speech_engine:
            self.status_var.set("Tizim hali yuklanmoqda...")
            return

        camera_id = self._selected_camera_id()
        try:
            self.pipeline = VideoPipeline(
                self.detector,
                DecisionLayer(near_threshold=0.40, mid_threshold=0.15),
                self.phrase_builder,
                self.speech_engine,
                camera_id=camera_id,
            )
        except Exception as exc:
            self.status_var.set(f"Kamera ochilmadi: {exc}")
            return

        self.running = True
        self.status_var.set("Navigatsiya faol")
        self.camera_status_var.set(f"Kamera {camera_id}")
        self.speech_engine.say("Navigatsiya ishga tushdi", priority=1)
        self._update_video()

    def stop(self):
        if not self.running and not self.pipeline:
            return
        self.running = False
        if self.pipeline:
            self.pipeline.release()
            self.pipeline = None
        self.status_var.set("To'xtatildi")
        self.phrase_var.set("Kutish rejimi")
        self.objects_var.set("0")
        self.level_var.set("--")
        self._draw_idle_canvas()

    def _update_video(self):
        if not self.running or not self.pipeline:
            return

        frame = self.pipeline.read_processed_frame()
        if frame is None:
            self.stop()
            self.status_var.set("Kamera signali yo'q")
            return

        self._draw_frame(frame)
        status = self.pipeline.get_status()
        self.phrase_var.set(status["phrase"])
        self.objects_var.set(str(status["objects"]))
        self.level_var.set(status["level"])

        self.root.after(15, self._update_video)

    def _draw_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(self.canvas.winfo_width(), 640)
        height = max(self.canvas.winfo_height(), 360)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(width // 2, height // 2, image=self.photo, anchor="center")

    def _draw_idle_canvas(self):
        if self.running:
            return
        width = max(self.canvas.winfo_width(), 640)
        height = max(self.canvas.winfo_height(), 360)
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, width, height, fill="#ffffff", outline="")
        self.canvas.create_text(
            width // 2,
            height // 2,
            text="Kamera kutish rejimida",
            fill="#98a2b3",
            font=("Segoe UI", 24, "bold"),
        )

    def close(self):
        self.stop()
        self.root.destroy()


def run_app():
    root = tk.Tk()
    app = BlindAssistantApp(root)
    app._draw_idle_canvas()
    root.mainloop()


if __name__ == "__main__":
    run_app()
