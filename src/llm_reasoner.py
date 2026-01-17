import json
import os
from google import genai
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Configure Gemini client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def reason_with_llm(features: dict):
    """
    Sends extracted ML features to Google's Gemini model and returns structured JSON.
    """

    feature_json = json.dumps(features, indent=2)

    prompt = f"""
You are an Image Quality & Suitability Reasoning System.

You are NOT given the raw image.
You are ONLY given machine-extracted ML signals.

Use ONLY the following features:

Extracted Features:
{feature_json}

Your job is to analyze:
- sharpness (blur_score; higher value = sharper image)
- lighting (brightness_score)
- clarity (color_contrast_score)
- background cleanliness (edge_density, background_color)
- content relevance (detected_objects)

Return ONLY a strict JSON object with:

{{
  "image_quality_score": number between 0 and 1,
  "issues_detected": [],
  "reasoning_summary": "1–2 sentences",
  "final_verdict": "Suitable" or "Not Suitable",
  "confidence": number between 0 and 1
}}

RULES:
- Only JSON.
- No markdown.
"""

    # Call Gemini model (use new official model id)
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config={
            "temperature": 0.2
        }
    )

    llm_output = response.text

    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        print("Invalid JSON from Gemini:\n", llm_output)
        return {
            "error": "invalid_json",
            "raw_output": llm_output
        }
