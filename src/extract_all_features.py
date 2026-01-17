from brightness_light_score import brightness_score
from color_contrast import contrast_score
from color_dominance import background_color
from detect_blur import detect_blur
from edge_density_clutter_detection import edge_density
from src.extract_objects import detect_objects

def extract_all_features(image_path):
    detect_object=detect_objects(image_path)
    brightness_Score = brightness_score(image_path)
    color_Contrast_Score = contrast_score(image_path)
    background_Color = background_color(image_path)
    blur_Score = detect_blur(image_path)
    edge_Density = edge_density(image_path)

    features={
        "detected_objects": detect_object,
        "brightness_score": brightness_Score,
        "color_contrast_score": color_Contrast_Score,
        "background_color": background_Color,
        "blur_score": blur_Score,
        "edge_density": edge_Density
    }

    return features


