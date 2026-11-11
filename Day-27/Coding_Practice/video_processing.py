import cv2
import os

# ==========================================
# 1. Define input and output paths
# ==========================================

input_path = r"C:\Users\Mr Waleed\OneDrive\Desktop\Day-27\Coding_Practice\input\travelling.mp4"
output_path = r"output\processed_edges.mp4"

os.makedirs("output", exist_ok=True)

# ==========================================
# 2. Read video
# ==========================================

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# ==========================================
# 3. Get video properties
# ==========================================

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\n========== VIDEO INFORMATION ==========")
print(f"FPS: {fps}")
print(f"Width: {width}")
print(f"Height: {height}")
print(f"Total Frames: {total_frames}")

if fps <= 0:
    fps = 30.0

# ==========================================
# 4. Initialize VideoWriter
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height),
    isColor=False
)

if not writer.isOpened():
    print("Error: Unable to create output video.")
    cap.release()
    exit()

# ==========================================
# 5. Read and process frames
# ==========================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video processing completed.")
        break

    frame_number += 1

    # Convert frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(gray, 100, 200)

    # Fit both previews in one window without changing the saved video size.
    preview_scale = min(1100 / (2 * width), 700 / height, 1.0)
    preview_size = (
        max(1, int(width * preview_scale)),
        max(1, int(height * preview_scale)),
    )
    original_preview = cv2.resize(
        frame, preview_size, interpolation=cv2.INTER_AREA
    )
    edges_preview = cv2.resize(
        edges, preview_size, interpolation=cv2.INTER_AREA
    )
    edges_preview = cv2.cvtColor(edges_preview, cv2.COLOR_GRAY2BGR)

    cv2.putText(
        original_preview, "Original", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    cv2.putText(
        edges_preview, "Canny Edges", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    preview = cv2.hconcat([original_preview, edges_preview])
    cv2.imshow("Original and Canny Edge Detection", preview)

    # Save processed frame
    writer.write(edges)

    print(f"Processing frame: {frame_number}/{total_frames}", end="\r")

    # Press Q to stop
    if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord('q'):
        break

# ==========================================
# 6. Release resources
# ==========================================

cap.release()
writer.release()

cv2.destroyAllWindows()

print("\nProcessed video saved successfully!")
print(f"Output: {output_path}")

