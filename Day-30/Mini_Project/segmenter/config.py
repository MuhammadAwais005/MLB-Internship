"""Paths and default settings for the segmentation tool."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs"

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# All methods run on a copy whose longest side is at most this many pixels.
# Thresholding does not need 12-megapixel detail, and it keeps every method fast.
# Final masks are scaled back up to the original resolution before saving.
WORK_SIZE = 1000

# The fixed value used by plain binary thresholding (the "guess" baseline).
FIXED_THRESHOLD = 127
