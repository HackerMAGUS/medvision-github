import re
import threading
import time
import traceback
import wave
import winsound
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime"
MMS_MODEL_ID = "facebook/mms-tts-uzb-script_cyrillic"


def _latin_uz_to_cyrillic(text):
    text = text.lower()
    text = re.sub(r"o[ʻ'`’]|\bo'", "ў", text)
    text = re.sub(r"g[ʻ'`’]|\bg'", "ғ", text)

    replacements = [
        ("sh", "ш"),
        ("ch", "ч"),
        ("ng", "нг"),
        ("yo", "ё"),
        ("yu", "ю"),
        ("ya", "я"),
        ("ye", "е"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)

    table = str.maketrans(
        {
            "a": "а",
            "b": "б",
            "d": "д",
            "e": "е",
            "f": "ф",
            "g": "г",
            "h": "ҳ",
            "i": "и",
            "j": "ж",
            "k": "к",
            "l": "л",
            "m": "м",
            "n": "н",
            "o": "о",
            "p": "п",
            "q": "қ",
            "r": "р",
            "s": "с",
            "t": "т",
            "u": "у",
            "v": "в",
            "x": "х",
            "y": "й",
            "z": "з",
        }
    )
    return text.translate(table)


class SpeechEngine:
    def __init__(self, cooldown=0.5, ttl=0.8, language="uz", rate=1.0):
        self.cooldown = cooldown
        self.ttl = ttl
        self.language = language
        self.rate = rate
        self.last_spoken = {}
        self.latest_request = None
        self.event = threading.Event()
        self._ready = threading.Event()

        self.worker_thread = threading.Thread(target=self._mms_worker, daemon=True)
        self.worker_thread.start()

    def _write_wav(self, path, sample_rate, audio):
        import numpy as np

        audio = np.asarray(audio, dtype=np.float32)
        max_val = np.max(np.abs(audio)) if audio.size else 0.0
        if max_val > 1.0:
            audio = audio / max_val
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = np.int16(audio * 32767.0)

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

    def _mms_worker(self):
        try:
            import torch
            from transformers import AutoTokenizer, VitsModel

            RUNTIME_DIR.mkdir(exist_ok=True)
            print(f"[SpeechEngine] Loading local Uzbek MMS-TTS: {MMS_MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(MMS_MODEL_ID)
            self.model = VitsModel.from_pretrained(MMS_MODEL_ID)
            self.model.eval()
            self.sample_rate = int(self.model.config.sampling_rate)
            self._ready.set()
            print(">>> [SpeechEngine] Uzbek MMS-TTS ready. <<<")

            while True:
                self.event.wait()
                self.event.clear()

                req = self.latest_request
                if req is None:
                    continue

                text, priority, timestamp = req
                if time.time() - timestamp > self.ttl:
                    print(f"[SpeechEngine] Phrase expired, skipping: {text}")
                    continue

                try:
                    synth_text = _latin_uz_to_cyrillic(text)
                    inputs = self.tokenizer(synth_text, return_tensors="pt")
                    if inputs.input_ids.numel() == 0:
                        print(f"[SpeechEngine] Empty tokenization, skipping: {text}")
                        continue

                    with torch.no_grad():
                        waveform = self.model(**inputs).waveform.squeeze().cpu().numpy()

                    temp_wav = RUNTIME_DIR / "temp_voice.wav"
                    self._write_wav(temp_wav, self.sample_rate, waveform)

                    if self.latest_request and self.latest_request[2] != timestamp and self.latest_request[1] > priority:
                        print("[SpeechEngine] Speech interrupted by higher priority phrase.")
                        continue

                    winsound.PlaySound(str(temp_wav), winsound.SND_FILENAME)

                except Exception as e:
                    err = f"Generation error: {e}\n{traceback.format_exc()}"
                    print(f"[SpeechEngine] {err}")

        except Exception as e:
            self._ready.set()
            err = f"Critical TTS error: {e}\n{traceback.format_exc()}"
            print(f"[SpeechEngine] {err}")

    def say(self, text, priority=0):
        if not text:
            return

        current_time = time.time()

        if priority == 0:
            last_time = self.last_spoken.get(text, 0)
            if current_time - last_time < self.cooldown:
                return

        self.last_spoken[text] = current_time

        if priority >= 1:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

        print(f"[TTS] New Uzbek phrase: {text} (priority: {priority})")
        self.latest_request = (text, priority, current_time)
        self.event.set()
