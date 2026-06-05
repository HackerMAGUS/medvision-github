import importlib

class PhraseBuilder:
    def __init__(self, language="ru"):
        self.language = language
        self.locale = self._load_locale(language)

    def _load_locale(self, lang):
        try:
            # Динамически импортируем модуль из папки locales
            return importlib.import_module(f"Backend.locales.{lang}")
        except ImportError:
            print(f"[PhraseBuilder] Язык '{lang}' не найден. Fallback to 'ru'")
            return importlib.import_module("Backend.locales.ru")

    def build_phrase(self, label, zone, distance):
        if not label:
            return ""
            
        # 1. Перевод объекта
        label_text = self.locale.LABELS.get(label, self.locale.LABELS.get("unknown"))
        
        # 2. Перевод зоны
        zone_text = self.locale.ZONES.get(zone, self.locale.ZONES.get("unknown"))
        
        # 3. Перевод дистанции (если нужно, можно отключать для far)
        distance_text = self.locale.DISTANCES.get(distance, "")
        
        # 4. Сборка фразы (вызов функции format_phrase из локали)
        phrase = self.locale.format_phrase(zone_text, label_text, distance_text)
        return phrase
