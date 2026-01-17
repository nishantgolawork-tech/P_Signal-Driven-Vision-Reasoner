import os
import json

from extract_all_features import extract_all_features
from llm_reasoner import reason_with_llm


def process_image(image_path, output_dir="outputs"):
    """
    Runs the full pipeline on a single image:
    - feature extraction
    - LLM reasoning
    - save JSON output
    """

    print(f"\n Processing image: {image_path}")

    # 1. Extract ML features
    features = extract_all_features(image_path)
    print(" Extracted Features:", features)

    if not features:
        print(" Feature extraction failed. Skipping image.")
        return

    # 2. LLM reasoning (Gemini)
    llm_result = reason_with_llm(features)
    print(" LLM Result:", llm_result)

    # 3. Prepare output path
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{image_name}_output.json")

    # 4. Save result
    with open(output_path, "w") as f:
        json.dump(
            {
                "image": image_path,
                "extracted_features": features,
                "llm_analysis": llm_result
            },
            f,
            indent=2
        )

    print(f" Output saved to: {output_path}")


def main():
    # List of images to process
    image_paths = [
        "examples/input1.jpg",
        "examples/input2.jpg",
        "examples/input3.jpg",
        "examples/input4.jpg",
        "examples/input5.webp",
        "examples/input6.jpg",
        
    ]

    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f" Image not found: {image_path}")
            continue

        process_image(image_path)


if __name__ == "__main__":
    main()
