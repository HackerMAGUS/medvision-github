import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Backend.decision_layer import DecisionLayer
from Backend.detector import Detector
from Backend.phrase_builder import PhraseBuilder
from Backend.speech_engine import SpeechEngine
from Frontend.app import run_app
from Frontend.video_pipeline import VideoPipeline


def run_console():
    print("=== Blind Assistant console mode ===")
    detector = Detector()
    decision_layer = DecisionLayer(near_threshold=0.40, mid_threshold=0.15)
    speech_engine = SpeechEngine(cooldown=2.0, ttl=8.0, language="uz")
    phrase_builder = PhraseBuilder(language="uz")
    speech_engine.say("Mahalliy navigatsiya tizimi ishga tushdi")

    pipeline = VideoPipeline(detector, decision_layer, phrase_builder, speech_engine, camera_id=0)
    pipeline.start()
    print("Program stopped.")


def main():
    if "--console" in sys.argv:
        run_console()
        return
    run_app()


if __name__ == "__main__":
    main()
