import cv2
def aspect_ratio(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    return round(w/h, 2)
