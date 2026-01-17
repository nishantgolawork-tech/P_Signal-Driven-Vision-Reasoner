import cv2
def edge_density(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    density = edges.mean() / 255  # normalize
    return round(float(density), 3)
