import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Backend.speech_engine import SpeechEngine


if __name__ == "__main__":
    tts = SpeechEngine(cooldown=0.1, ttl=5.0, language="uz")
    tts._ready.wait(timeout=30)
    tts.say("Oldinda odam, yaqin", priority=1)
    time.sleep(5)
    print("DONE")
