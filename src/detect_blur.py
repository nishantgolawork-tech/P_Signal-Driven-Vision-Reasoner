import cv2

def detect_blur(image_path):
    # Read image in grayscale (faster + better for blur detection)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # If image fails to load
    if image is None:
        return None
    
    # Laplacian operator (measures edges)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    
    # Higher value = sharper image
    return round(float(laplacian_var), 3)
