
from ultralytics import YOLO
from pathlib import Path

# 1. Load pretrained YOLO11 nano model
print("Loading YOLO11n model...")
model = YOLO("yolo11n.pt")

# 2. Define test dataset directory
test_images = Path(r"C:\Users\92348\Desktop\MLBench\Day-23\PPE_Detection_YOLO\test\images")

# 3. Define output directory
output_dir = Path(r"C:\Users\92348\Desktop\MLBench\Day-23\PPE_Detection_YOLO\outputs")

# 4. Check dataset
if not test_images.exists():
    raise FileNotFoundError(f"Dataset folder not found: {test_images}")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
images = [
    image for image in test_images.iterdir()
    if image.suffix.lower() in image_extensions
]

print(f"\nTotal test images: {len(images)}")

# 5. Run inference
print("\nRunning YOLO inference...")

results = model.predict(
    source=str(test_images),
    conf=0.25,
    save=True,
    project=str(output_dir),
    name="ppe_detection",
    exist_ok=True,
    verbose=False
)

# 6. Display prediction information
total_detections = 0

print("\n========== DETECTION RESULTS ==========")

for result in results:

    image_name = Path(result.path).name
    boxes = result.boxes

    print(f"\nImage: {image_name}")
    print(f"Objects detected: {len(boxes)}")

    total_detections += len(boxes)

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])

        print(
            f"  Class: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )

# 7. Final summary
print("\n========== FINAL SUMMARY ==========")
print(f"Images processed: {len(results)}")
print(f"Total detections: {total_detections}")
print(f"Average detections per image: "
      f"{total_detections / max(len(results), 1):.2f}")

print("\nAnnotated images saved in:")
print(output_dir / "ppe_detection")
