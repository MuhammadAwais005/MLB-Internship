
from ultralytics import YOLO
from pathlib import Path

# 1. Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# 2. Select your image
image_path = r"C:\Users\92348\Desktop\MLBench\Day-23\images\1.jpg"

# 3. Perform object detection
results = model.predict(
    source=image_path,
    conf=0.35,
    save=True,
    project="outputs",
    name="practice2",
    exist_ok=True
)

# 4. Analyze predictions
for result in results:

    print("\nIMAGE:", Path(result.path).name)

    print("\nDETECTED OBJECTS:")

    for box in result.boxes:

        # Class information
        class_id = int(box.cls[0])
        class_name = result.names[class_id]

        # Confidence score
        confidence = float(box.conf[0])

        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Bounding box dimensions
        width = x2 - x1
        height = y2 - y1

        print("--------------------------")

        print("Object:", class_name)
        print("Class ID:", class_id)
        print("Confidence:", round(confidence, 3))

        print("Bounding Box:")
        print("X1:", round(x1, 2))
        print("Y1:", round(y1, 2))
        print("X2:", round(x2, 2))
        print("Y2:", round(y2, 2))

        print("Width:", round(width, 2))
        print("Height:", round(height, 2))

print("\nPractice 2 completed!")
