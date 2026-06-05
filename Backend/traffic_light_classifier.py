import cv2
import numpy as np

class TrafficLightClassifier:
    def __init__(self, model_path=None):
        """
        [PLACEHOLDER INTERFACE]
        Для прототипа используется HSV/Эвристический анализ Crop-изображения.
        В будущем: загрузка MobileNetV2/ResNet для анализа crop (red, yellow, green, walk, dont_walk).
        """
        pass

    def classify(self, crop, tl_type):
        if crop.size == 0:
            return "unknown"
            
        # Отбрасываем слишком маленькие светофоры (невозможно надежно определить HSV)
        if crop.shape[0] < 20 or crop.shape[1] < 10:
            return "unknown"
            
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        
        # Определяем базовые пороги цветов
        mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_yellow = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([90, 255, 255]))
        
        # Белый человечек пешеходного (высокая яркость, низкая насыщенность)
        mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255])) 

        r_cnt = cv2.countNonZero(mask_red)
        y_cnt = cv2.countNonZero(mask_yellow)
        g_cnt = cv2.countNonZero(mask_green)
        w_cnt = cv2.countNonZero(mask_white)
        
        total = crop.shape[0] * crop.shape[1]
        
        # --- ЛОГИКА АВТОМОБИЛЬНОГО ---
        if tl_type == "vehicle":
            if r_cnt > y_cnt and r_cnt > g_cnt and r_cnt > total*0.015:
                return "red"
            elif y_cnt > r_cnt and y_cnt > g_cnt and y_cnt > total*0.015:
                return "yellow"
            elif g_cnt > r_cnt and g_cnt > y_cnt and g_cnt > total*0.015:
                return "green"
            return "unknown"
            
        # --- ЛОГИКА ПЕШЕХОДНОГО ---
        elif tl_type == "pedestrian":
            if r_cnt > w_cnt and r_cnt > g_cnt and r_cnt > total*0.015:
                return "dont_walk" # Красный человек
            if g_cnt > r_cnt and g_cnt > w_cnt and g_cnt > total*0.015:
                return "walk"      # Зеленый человек
            if w_cnt > r_cnt and w_cnt > g_cnt and w_cnt > total*0.015:
                return "walk"      # Белый человек (разрешающий)
            return "unknown"
            
        return "unknown"
