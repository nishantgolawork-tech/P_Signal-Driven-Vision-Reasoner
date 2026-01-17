import cv2
import numpy as np

def brightness_score(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = hsv[..., 2].mean()
    return round(float(brightness), 2)
