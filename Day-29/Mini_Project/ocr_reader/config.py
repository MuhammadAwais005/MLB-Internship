"""Project-wide paths and default settings, kept in one place."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TEXT_DIR = OUTPUT_DIR / "text"
PREVIEW_DIR = OUTPUT_DIR / "previews"
SUMMARY_FILE = OUTPUT_DIR / "summary.csv"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# EasyOCR language codes. "en" is enough for this project; you can add more,
# e.g. ["en", "ur"] for English + Urdu.
DEFAULT_LANGUAGES = ["en"]

# Small images are upscaled before OCR (only when "enhance" is on), because
# EasyOCR misses letters that are just a few pixels tall.
MIN_IMAGE_SIZE = 1000
