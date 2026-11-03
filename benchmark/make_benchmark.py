import json
from pathlib import Path
from PIL import Image, ImageDraw

from config import (
    BENCHMARK_DIR,
    BENCHMARK_FILES_DIR,
    GROUND_TRUTH_PATH,
)

def create_text_fixtures(output_dir: Path) -> dict:

    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "exact_original.txt": "ClutterIQ benchmark: identical content.",
        "exact_copy.txt": "ClutterIQ benchmark: identical content.",
        "unique_document.txt": "This is a unique benchmark document.",
    }

    for filename, content in files.items():
        file_path = output_dir / filename
        file_path.write_text(content, encoding="utf-8")

    ground_truth = {
        "exact_duplicates": [
            ["exact_original.txt", "exact_copy.txt"]
        ],
        "near_duplicates": [],
        "unique_files": [
            "unique_document.txt"
        ],
    }

    return ground_truth

def create_image_fixtures(output_dir: Path) -> dict:
    

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a synthetic 256x256 image
    original = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(original)

    draw.rectangle(
        (30, 30, 220, 220),
        fill="lightblue"
    )

    draw.ellipse(
        (75, 75, 180, 180),
        fill="orange"
    )

    draw.line(
        (30, 220, 220, 30),
        fill="black",
        width=5
    )

    original_path = output_dir / "photo_original.png"
    resized_path = output_dir / "photo_resized.png"

    # Save original image
    original.save(original_path)

    # Create a resized version
    resized = original.resize(
        (128, 128),
        Image.Resampling.LANCZOS
    )

    resized.save(resized_path)

    return {
        "near_duplicates": [
            ["photo_original.png", "photo_resized.png"]
        ]
    }

def generate_benchmark() -> dict:

    BENCHMARK_FILES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate text fixtures
    text_truth = create_text_fixtures(BENCHMARK_FILES_DIR)

    # Generate image fixtures
    image_truth = create_image_fixtures(BENCHMARK_FILES_DIR)

    # Combine ground truth
    ground_truth = {
        "exact_duplicates": text_truth["exact_duplicates"],
        "near_duplicates": image_truth["near_duplicates"],
        "unique_files": text_truth["unique_files"],
    }

    # Save manifest
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    with GROUND_TRUTH_PATH.open("w", encoding="utf-8") as file:
        json.dump(ground_truth, file, indent=4)

    return ground_truth


if __name__ == "__main__":
    result = generate_benchmark()
    print("Benchmark generated successfully!")
    print(json.dumps(result, indent=4))