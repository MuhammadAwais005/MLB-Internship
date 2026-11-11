
import cv2
import os

from processing_utils import process_frame, display_frames


def process_video(input_path, output_path):

    # Open input video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Unable to open input video.")
        return

    # Retrieve video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    print("\n========== VIDEO INFORMATION ==========")
    print(f"FPS: {fps}")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Total Frames: {total_frames}")

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

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
        return

    frame_count = 0

    print("\nProcessing video... Press Q to stop.")

    try:

        while True:

            # Read next frame
            ret, frame = cap.read()

            if not ret:
                print("\nVideo processing completed.")
                break

            frame_count += 1

            # Apply processing pipeline
            gray, blurred, edges = process_frame(frame)

            # Display original and processed frames
            display_frames(frame, edges)

            # Save processed frame
            writer.write(edges)

            print(
                f"Processing: {frame_count}/{total_frames}",
                end="\r"
            )

            # Maintain approximate original playback speed
            delay = max(1, int(1000 / fps))

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                print("\nProcessing stopped by user.")
                break

    finally:

        # Release resources
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"\nProcessed video saved at: {output_path}")
