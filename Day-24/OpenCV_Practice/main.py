
import cv2
import os
import numpy as np

# =====================================
# 1. READ IMAGE AND DISPLAY PROPERTIES
# =====================================

image_path = r"C:\Users\92348\Desktop\MLBench\Day-24\OpenCV_Practice\Images\panda.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")

height, width = image.shape[:2]
channels = image.shape[2] if len(image.shape) > 2 else 1

file_size = os.path.getsize(image_path)

print("========== IMAGE INFORMATION ==========")
print("Height:", height)
print("Width:", width)
print("Channels:", channels)
print("Dimensions:", image.shape)
print("File Size:", round(file_size / 1024, 2), "KB")


# Create output directory
os.makedirs("output", exist_ok=True)


# =====================================
# 2. CONVERT COLOR IMAGE TO GRAYSCALE
# =====================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("output/grayscale.jpg", gray)


# =====================================
# 3. RESIZE TO DIFFERENT RESOLUTIONS
# =====================================

resize_640 = cv2.resize(image, (640, 640))
resize_320 = cv2.resize(image, (320, 320))
resize_1280 = cv2.resize(image, (1280, 720))

cv2.imwrite("output/resized_640.jpg", resize_640)
cv2.imwrite("output/resized_320.jpg", resize_320)
cv2.imwrite("output/resized_1280.jpg", resize_1280)


# =====================================
# 4. CROP DIFFERENT REGIONS
# =====================================

crop_1 = image[0:200, 0:200]
crop_2 = image[100:400, 100:400]
crop_3 = image[200:500, 200:600]

cv2.imwrite("output/crop_1.jpg", crop_1)
cv2.imwrite("output/crop_2.jpg", crop_2)
cv2.imwrite("output/crop_3.jpg", crop_3)


# =====================================
# 5. ROTATE IMAGE
# =====================================

rotate_90 = cv2.rotate(
    image, cv2.ROTATE_90_CLOCKWISE
)

rotate_180 = cv2.rotate(
    image, cv2.ROTATE_180
)

rotate_270 = cv2.rotate(
    image, cv2.ROTATE_90_COUNTERCLOCKWISE
)

cv2.imwrite("output/rotate_90.jpg", rotate_90)
cv2.imwrite("output/rotate_180.jpg", rotate_180)
cv2.imwrite("output/rotate_270.jpg", rotate_270)


# =====================================
# 6. FLIP IMAGE
# =====================================

horizontal = cv2.flip(image, 1)
vertical = cv2.flip(image, 0)

cv2.imwrite("output/horizontal_flip.jpg", horizontal)
cv2.imwrite("output/vertical_flip.jpg", vertical)


# =====================================
# 7. DRAW SHAPES AND ADD TEXT
# =====================================

annotated = image.copy()

# Rectangle
cv2.rectangle(
    annotated,
    (100, 100),
    (300, 250),
    (0, 255, 0),
    3
)

# Circle
cv2.circle(
    annotated,
    (400, 200),
    50,
    (0, 0, 255),
    3
)

# Line
cv2.line(
    annotated,
    (50, 300),
    (500, 300),
    (255, 0, 0),
    3
)

# Polygon
points = [
    [500, 100],
    [600, 200],
    [550, 350],
    [450, 250]
]



points = np.array(points, dtype=np.int32)

cv2.polylines(
    annotated,
    [points],
    True,
    (255, 255, 0),
    3
)

# Custom text
name = "Muhammad Awais"
date = "04 November 2026"

cv2.putText(
    annotated,
    name,
    (50, 400),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 255),
    2
)

cv2.putText(
    annotated,
    date,
    (50, 450),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

cv2.imwrite("output/annotated_image.jpg", annotated)


# =====================================
# FINAL RESULT
# =====================================

print("\nAll operations completed successfully!")
print("Check the output folder for processed images.")

cv2.imshow("Original Image", image)
cv2.imshow("Annotated Image", annotated)
cv2.imshow("Grayscale Image", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
