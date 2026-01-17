import cv2
def contrast_score(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contrast = img.std()
    return round(float(contrast), 2)
