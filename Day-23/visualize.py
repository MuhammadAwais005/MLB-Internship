
from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np

# --------------------------------
# 1. Load YOLO model
# --------------------------------
model = YOLO("yolo11n.pt")

image_folder = Path(r"C:\Users\92348\Desktop\MLBench\Day-23\images")
output_folder = Path(r"C:\Users\92348\Desktop\MLBench\Day-23\outputs\visualizations")
output_folder.mkdir(parents=True, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

image_files = [
    p for p in image_folder.iterdir()
    if p.suffix.lower() in image_extensions
]

# --------------------------------
# 2. Process each image
# --------------------------------
for image_path in image_files:

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path)
        continue

    height, width = image.shape[:2]

    results = model.predict(
        source=str(image_path),
        conf=0.35,
        verbose=False
    )

    # Prepare measurement panel
    detections = results[0].boxes
    panel_width = 440
    panel_height = max(height, 150 + len(detections) * 125)

    panel = np.zeros(
        (panel_height, panel_width, 3),
        dtype=np.uint8
    )

    panel[:] = (35, 35, 35)

    # --------------------------------
    # 3. Panel heading
    # --------------------------------
    cv2.putText(
        panel,
        "YOLO DETECTION MEASUREMENTS",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        panel,
        f"Image: {image_path.name}",
        (15, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        panel,
        f"Image Size: {width} x {height}",
        (15, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )

    cv2.putText(
        panel,
        f"Total Detections: {len(detections)}",
        (15, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    # --------------------------------
    # 4. Draw bounding boxes
    # --------------------------------
    for i, box in enumerate(detections):

        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        x1, y1, x2, y2 = map(
            lambda v: int(round(v)),
            (x1, y1, x2, y2)
        )

        box_width = x2 - x1
        box_height = y2 - y1

        # Draw bounding rectangle
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        cv2.circle(
            image,
            (center_x, center_y),
            4,
            (0, 0, 255),
            -1
        )

        # Draw class and confidence
        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # --------------------------------
        # 5. Display measurements
        # --------------------------------
        start_y = 180 + i * 125

        details = [
            f"Object {i + 1}: {class_name}",
            f"Class ID: {class_id}",
            f"Confidence: {confidence:.3f}",
            f"X1: {x1}    Y1: {y1}",
            f"X2: {x2}    Y2: {y2}",
            f"Width: {box_width} px",
            f"Height: {box_height} px"
        ]

        for j, line in enumerate(details):

            cv2.putText(
                panel,
                line,
                (15, start_y + j * 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1
            )

    # --------------------------------
    # 6. Combine image and panel
    # --------------------------------
    canvas = np.zeros(
        (panel_height, width + panel_width, 3),
        dtype=np.uint8
    )

    canvas[:height, :width] = image
    canvas[:, width:] = panel

    # --------------------------------
    # 7. Save visualization
    # --------------------------------
    output_path = output_folder / f"{image_path.stem}_visualized.jpg"

    cv2.imwrite(str(output_path), canvas)

    print("Saved:", output_path)

    # Display result
    cv2.imshow("YOLO Detection Measurements", canvas)

    print("Press any key to view the next image...")
    cv2.waitKey(0)

cv2.destroyAllWindows()

print("\nAll visualizations completed!")
