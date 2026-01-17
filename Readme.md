# Image Quality Assessment System

A comprehensive machine learning pipeline that extracts visual features from images and uses Google's Gemini AI to perform intelligent quality assessment and suitability analysis.

## 🎯 Project Overview

This project automates the evaluation of image quality through a two-stage process:

1. **Feature Extraction**: Extract machine learning signals from images including blur detection, brightness, color contrast, object detection, and edge density
2. **LLM Analysis**: Feed extracted features to Google's Gemini 2.5 Flash model for intelligent reasoning about image quality and suitability

The system is designed to assess whether images are suitable for use in professional applications based on multiple visual quality metrics.

## 📁 Project Structure

```
P_Signal Driven Vision Reasoner/
├── src/                              # Main source code directory
│   ├── main.py                      # Entry point - processes images through pipeline
│   ├── extract_all_features.py      # Orchestrates feature extraction
│   ├── llm_reasoner.py              # Integrates with Google Gemini API
│   │
│   ├── Feature Extraction Modules:
│   ├── detect_blur.py               # Laplacian variance-based blur detection
│   ├── brightness_light_score.py    # HSV-based brightness analysis
│   ├── color_contrast.py            # Standard deviation-based contrast scoring
│   ├── color_dominance.py           # Average RGB color calculation
│   ├── edge_density_clutter_detection.py  # Canny edge detection for clutter
│   ├── extract_objects.py           # YOLOv8 object detection
│   │
│   ├── Utility Modules:
│   ├── aspect_ratio.py              # Image aspect ratio calculation
│   ├── exif_meta_information.py     # EXIF metadata extraction
│   ├── list_all_models.py           # YOLO model utilities
│   │
│   ├── Testing:
│   ├── test_all_feature_extracted.py
│   ├── test_aspect_ratio.py
│   ├── test_blur.py
│   ├── test_brightness_score.py
│   ├── test_color_contras.py
│   ├── test_color_dominance.py
│   ├── test_edge_density_clutter_detection.py
│   ├── test_exif_meta_information.py
│   ├── test_llm_reasoner.py
│   └── test.py
│
├── examples/                         # Sample images for testing
│   ├── input1.jpg
│   ├── input2.jpg
│   ├── input3.jpg
│   ├── input4.jpg
│   ├── input5.webp
│   └── input6.jpg
│
├── outputs/                          # Generated JSON analysis results
│   └── [image_name]_output.json
│
├── yolov8s.pt                        # YOLOv8 Small pre-trained model
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables (API keys)
└── Readme.md                         # This file
```

## 🚀 Features

### Image Quality Metrics Extracted

| Feature | Module | Description |
|---------|--------|-------------|
| **Blur Detection** | `detect_blur.py` | Laplacian variance (higher = sharper) |
| **Brightness Score** | `brightness_light_score.py` | HSV Value channel mean (0-255) |
| **Color Contrast** | `color_contrast.py` | Standard deviation of grayscale values |
| **Background Color** | `color_dominance.py` | Average RGB values (list of 3 floats) |
| **Edge Density** | `edge_density_clutter_detectin.py` | Normalized Canny edge density (0-1) |
| **Object Detection** | `extract_objects.py` | YOLOv8 detected object labels |
| **Aspect Ratio** | `aspect_ratio.py` | Image width-to-height ratio |
| **EXIF Metadata** | `exif_meta_information.py` | Camera settings, GPS, timestamps |

### LLM Analysis Output

The Gemini API returns a structured JSON assessment:

```json
{
  "image_quality_score": 0.85,
  "issues_detected": ["slight_blur", "low_contrast"],
  "reasoning_summary": "Image has good lighting but minor focus issues.",
  "final_verdict": "Suitable",
  "confidence": 0.92
}
```

## 📦 Requirements

- **Python 3.8+**
- **OpenCV** (cv2) - Image processing
- **PIL/Pillow** - Image manipulation
- **NumPy** - Numerical operations
- **Ultralytics** - YOLOv8 object detection
- **google-genai** - Google Gemini API (modern version)
- **python-dotenv** - Environment variable management

### All dependencies are listed in `requirements.txt`

## 🔧 Installation

### 1. Clone/Setup the Project
```bash
cd "P_Artikate Studio task"
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey)

### 5. Verify YOLOv8 Model

Ensure `yolov8s.pt` exists in the project root. If not, it will be downloaded automatically on first use.

## 🎯 Usage

### Process Images via Command Line

```bash
python src/main.py
```

This will process all images in the `examples/` directory and save results to `outputs/`

### Process a Single Image Programmatically

```python
from src.main import process_image

process_image("path/to/image.jpg", output_dir="outputs")
```

### Extract Features Only

```python
from src.extract_all_features import extract_all_features

features = extract_all_features("path/to/image.jpg")
print(features)
```

Output:
```python
{
    "detected_objects": ["car", "person"],
    "brightness_score": 145.32,
    "color_contrast_score": 62.45,
    "background_color": [198.5, 176.3, 182.7],
    "blur_score": 2100.50,
    "edge_density": 0.087
}
```

### Get LLM Analysis

```python
from src.extract_all_features import extract_all_features
from src.llm_reasoner import reason_with_llm

features = extract_all_features("path/to/image.jpg")
analysis = reason_with_llm(features)
print(analysis)
```

## 🧪 Testing

Run individual feature tests:

```bash
# Test blur detection
python src/test_blur.py

# Test brightness scoring
python src/test_brightness_score.py

# Test color contrast
python src/test_color_contras.py

# Test color dominance
python src/test_color_dominance.py

# Test edge density
python src/test_edge_density_clutter_detection.py

# Test all features extraction
python src/test_all_feature_extracted.py

# Test LLM reasoner
python src/test_llm_reasoner.py
```

## 📊 Output Format

Each processed image generates a JSON file in `outputs/` with the following structure:

```json
{
  "image": "examples/input1.jpg",
  "extracted_features": {
    "detected_objects": ["car"],
    "brightness_score": 213.64,
    "color_contrast_score": 84.54,
    "background_color": [202.55, 186.21, 189.66],
    "blur_score": 2069.27,
    "edge_density": 0.099
  },
  "llm_analysis": {
    "image_quality_score": 0.82,
    "issues_detected": ["moderate_blur", "high_edge_density"],
    "reasoning_summary": "Image has acceptable brightness but shows motion blur.",
    "final_verdict": "Suitable",
    "confidence": 0.88
  }
}
```

## 🔑 Key Components

### `main.py`
- Orchestrates the complete pipeline
- Processes multiple images
- Saves structured JSON outputs
- Handles file I/O and error logging

### `extract_all_features.py`
- Central hub for all feature extraction
- Calls individual feature modules
- Returns consolidated feature dictionary

### `llm_reasoner.py`
- Interfaces with Google Gemini 2.5 Flash API
- Sends features as structured prompts
- Parses and validates JSON responses
- Error handling for API failures

### Feature Extraction Modules
Each module is self-contained and can be used independently:
- Takes image path as input
- Returns single computed metric or list
- Uses appropriate CV2/PIL techniques

## ⚙️ Configuration

### Model Settings (in `llm_reasoner.py`)
```python
response = client.models.generate_content(
    model="gemini-2.5-flash",  # Model selection
    contents=prompt,
    config={"temperature": 0.2}  # Low temperature for consistency
)
```

### YOLO Settings (in `extract_objects.py`)
```python
yolo_model = YOLO("yolov8s.pt")  # Small model for speed
detect_objects(image_path, conf_threshold=0.4)  # Confidence threshold
```

## 🐛 Troubleshooting

### ModuleNotFoundError for google-genai
```bash
pip install google-genai
```

### API Key Issues
- Verify `.env` file exists in project root
- Check API key is valid in [Google AI Studio](https://aistudio.google.com/apikey)
- Ensure GEMINI_API_KEY is set correctly

### YOLOv8 Model Not Found
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

### Image Processing Errors
- Ensure image format is supported (JPG, PNG, WEBP)
- Check image file exists and is readable
- Verify image is not corrupted

## 📈 Performance Notes

- **YOLOv8 Small**: ~50-100ms per image on CPU
- **Feature Extraction**: ~300-500ms per image
- **LLM API Call**: ~1-3 seconds (depends on network)
- **Total Pipeline**: ~2-5 seconds per image

## 🔄 API Usage

Uses Google's Gemini 2.5 Flash model with:
- Temperature: 0.2 (for deterministic outputs)
- Max tokens: Auto
- JSON mode: Enforced via prompt instructions

## 📝 Notes

- The system prioritizes **reproducibility** with low LLM temperature
- Feature extraction is **parallelizable** for batch processing
- All outputs are **JSON-based** for easy integration
- The project uses **YOLOv8 Small** for speed over larger models



## 👤 Author

Nishant Gola

---

**Last Updated**: January 2026
