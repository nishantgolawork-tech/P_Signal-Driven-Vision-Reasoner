from PIL import Image
import numpy as np

def background_color(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    avg_color = img_np.mean(axis=(0,1))   # RGB mean
    return avg_color.tolist()
