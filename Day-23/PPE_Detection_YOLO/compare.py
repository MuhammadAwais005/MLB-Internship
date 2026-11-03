
from ultralytics import YOLO
from pathlib import Path
import cv2
import yaml
import numpy as np

# Load pretrained model
model = YOLO("yolo11n.pt")

# Dataset paths
image_dir = Path("test/images")
label_dir = Path("test/labels")
output_dir = Path("outputs/comparisons")
output_dir.mkdir(parents=True, exist_ok=True)

# Load class names
with open("data.yaml", "r") as file:
    data = yaml.safe_load(file)

names = data["names"]

# Handle either list or dictionary class names
if isinstance(names, list):
    class_names = names
else:
    class_names = {int(k): v for k, v in names.items()}

# Select first five test images
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
images = [
    p for p in sorted(image_dir.iterdir())
    if p.suffix.lower() in image_extensions
][:5]

for image_path in images:

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]

    # LEFT: Ground Truth
    ground_truth = image.copy()

    label_path = label_dir / f"{image_path.stem}.txt"

    if label_path.exists():
        with open(label_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])
            x_center, y_center, box_w, box_h = map(float, values[1:])

            x1 = int((x_center - box_w / 2) * width)
            y1 = int((y_center - box_h / 2) * height)
            x2 = int((x_center + box_w / 2) * width)
            y2 = int((y_center + box_h / 2) * height)

            class_name = class_names[class_id]

            cv2.rectangle(
                ground_truth, (x1, y1), (x2, y2),
                (255, 0, 0), 2
            )

            cv2.putText(
                ground_truth, class_name,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2
            )

    # RIGHT: YOLO Predictions
    results = model.predict(
        source=str(image_path),
        conf=0.25,
        verbose=False
    )

    prediction = results[0].plot()

    # Add headings
    cv2.putText(
        ground_truth, "GROUND TRUTH",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (255, 0, 0), 2
    )

    cv2.putText(
        prediction, "YOLO PREDICTION",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (0, 255, 0), 2
    )

    # Combine both images
    comparison = np.hstack((ground_truth, prediction))

    output_path = output_dir / f"{image_path.stem}_comparison.jpg"

    cv2.imwrite(str(output_path), comparison)

    print("Saved:", output_path)

print("\nGround Truth vs Prediction comparison completed!")
