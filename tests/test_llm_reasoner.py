from src.llm_reasoner import reason_with_llm

features = {
    "detected_objects": ["car"],
    "brightness_score": 213.64,
    "color_contrast_score": 84.54,
    "background_color": [202.55, 186.21, 189.66],
    "blur_score": 2069.27,
    "edge_density": 0.099
}

print(reason_with_llm(features))
