
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO


# Get the directory where this Python file exists
BASE_DIR = Path(__file__).resolve().parent


def resize_for_display(frame, max_width=1280, max_height=720):
    """Resize only the live preview while keeping the saved frame at full resolution."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale >= 1.0:
        return frame

    display_width = max(1, int(width * scale))
    display_height = max(1, int(height * scale))
    return cv2.resize(
        frame,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA
    )


def main(source):

    # --------------------------------
    # 1. Load YOLO Model
    # --------------------------------

    print("Loading YOLO model...")

    model = YOLO("yolo11n.pt")

    print("YOLO model loaded successfully!")

    # --------------------------------
    # 2. Resolve Video Source
    # --------------------------------

    if isinstance(source, int):
        video_source = source
        output_name = "webcam_output.mp4"

    else:
        video_source = Path(source)

        if not video_source.is_absolute():
            video_source = BASE_DIR / video_source

        output_name = "people_detection_output.mp4"

        if not video_source.exists():
            print(f"Error: Video not found: {video_source}")
            return

    # --------------------------------
    # 3. Open Video Source
    # --------------------------------

    cap = cv2.VideoCapture(str(video_source))

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    if width <= 0 or height <= 0:
        print("Error: Invalid video dimensions.")
        cap.release()
        return

    print(f"Video Resolution: {width}x{height}")
    print(f"FPS: {fps}")

    # --------------------------------
    # 4. Create Output Directory
    # --------------------------------

    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / output_name

    # --------------------------------
    # 5. Initialize Video Writer
    # --------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("Error: Could not initialize video writer.")
        cap.release()
        return

    print("\nPeople Detection & Tracking Started!")
    print("Press Q to stop processing.\n")

    cv2.namedWindow(
        "YOLO People Detection & Tracking",
        cv2.WINDOW_NORMAL
    )

    frame_number = 0

    # --------------------------------
    # 6. Process Video Frames
    # --------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video processing completed.")
            break

        frame_number += 1

        # --------------------------------
        # 7. YOLO Detection + ByteTrack
        # --------------------------------

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],       # Class 0 = Person
            conf=0.35,
            verbose=False
        )

        result = results[0]

        # Number of people detected in current frame
        people_count = 0

        if result.boxes is not None:

            boxes = result.boxes

            people_count = len(boxes)

            # --------------------------------
            # 8. Draw Bounding Boxes and IDs
            # --------------------------------

            for box in boxes:

                # Bounding box coordinates
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # Detection confidence
                confidence = float(box.conf[0])

                # Tracking ID
                track_id = None

                if box.id is not None:
                    track_id = int(box.id[0])

                # Prepare label
                if track_id is not None:
                    label = f"ID: {track_id} | Conf: {confidence:.2f}"
                else:
                    label = f"Person | Conf: {confidence:.2f}"

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw label
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # --------------------------------
        # 9. Display Live People Count
        # --------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (300, 60),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"People Count: {people_count}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # --------------------------------
        # 10. Display Frame Number
        # --------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # --------------------------------
        # 11. Save Processed Frame
        # --------------------------------

        writer.write(frame)

        # --------------------------------
        # 12. Display Processed Video
        # --------------------------------

        cv2.imshow(
            "YOLO People Detection & Tracking",
            resize_for_display(frame)
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Processing stopped by user.")
            break

    # --------------------------------
    # 13. Release Resources
    # --------------------------------

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\n--------------------------------")
    print("PROCESSING FINISHED")
    print("--------------------------------")
    print(f"Total frames processed: {frame_number}")
    print(f"Output saved at: {output_path}")


# ------------------------------------------------
# Image Testing Function (Optional)
#------------------------------------------------

def process_image(image_path):

    model = YOLO("yolo11n.pt")

    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    frame = cv2.imread(str(image_path))

    if frame is None:
        print("Could not read image.")
        return

    # Detect people only
    results = model.predict(
        source=frame,
        classes=[0],
        conf=0.35,
        verbose=False
    )

    result = results[0]

    people_count = 0

    if result.boxes is not None:

        people_count = len(result.boxes)

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])

            label = f"Person: {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        f"People Count: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "image_detection_output.jpg"

    cv2.imwrite(str(output_path), frame)

    cv2.namedWindow(
        "YOLO Image People Detection",
        cv2.WINDOW_NORMAL
    )
    cv2.imshow(
        "YOLO Image People Detection",
        resize_for_display(frame)
    )

    print(f"People detected: {people_count}")
    print(f"Output saved: {output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()



# --------------------------------
# Main Entry Point
# --------------------------------


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        default="input/badminton.mp4"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "image"],
        default="video"
    )

    args = parser.parse_args()

    if args.mode == "image":
        process_image(args.source)

    else:
        source = int(args.source) if args.source.isdigit() else args.source
        main(source)

