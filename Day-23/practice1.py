
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# 1. Check input folder
# -----------------------------
image_folder = Path(r"C:\Users\92348\Desktop\MLBench\Day-23\images")
output_folder = Path(r"C:\Users\92348\Desktop\MLBench\Day-23\outputs")

if not image_folder.exists():
    raise FileNotFoundError(
        "The 'images' folder was not found."
    )

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_files = [
    path for path in image_folder.iterdir()
    if path.is_file()
    and path.suffix.lower() in image_extensions
]

if not image_files:
    raise FileNotFoundError(
        "No supported images found inside 'images'."
    )

# -----------------------------
# 2. Load pretrained model
# -----------------------------
model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully!")
print(f"Images found: {len(image_files)}")

# -----------------------------
# 3. Single-image detection
# -----------------------------
single_image = image_files[0]

print(f"\nDetecting objects in: {single_image}")

single_results = model.predict(
    source=str(single_image),
    conf=0.35,
    save=True,
    project=str(output_folder),
    name="single_image",
    exist_ok=True
)

for result in single_results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]

        print(
            f"Detected: {class_name} "
            f"| Confidence: {confidence:.2f}"
        )

# -----------------------------
# 4. Multiple-image detection
# -----------------------------
print("\nProcessing multiple images...")

batch_results = model.predict(
    source=str(image_folder),
    conf=0.35,
    save=True,
    project=str(output_folder),
    name="multiple_images",
    exist_ok=True
)

for result in batch_results:
    print(f"\nImage: {Path(result.path).name}")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]

        print(
            f"  {class_name}: {confidence:.2f}"
        )

# -----------------------------
# 5. Summary
# -----------------------------
print("\nPractice 1 completed!")
print(f"Single-image results: {output_folder / 'single_image'}")
print(f"Multiple-image results: {output_folder / 'multiple_images'}")
