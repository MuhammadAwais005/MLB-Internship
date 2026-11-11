
import cv2
import os

from processing_utils import process_frame, display_frames


def process_webcam(output_path):

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        return

    print("\nWebcam started successfully!")
    print("Press Q to stop recording.")

    # Retrieve webcam properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    writer = None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames_saved = 0

    try:

        while True:

            # Capture live frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to capture webcam frame.")
                break

            # Apply processing pipeline
            gray, blurred, edges = process_frame(frame)

            # Initialize writer using actual frame dimensions
            if writer is None:

                height, width = edges.shape

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (width, height),
                    isColor=False
                )

                if not writer.isOpened():
                    print("Error: Unable to create webcam output.")
                    break

            # Display frames
            display_frames(frame, edges)

            # Save processed webcam frame
            writer.write(edges)
            frames_saved += 1

            # Exit on Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        # Release camera and writer
        cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    if frames_saved > 0:
     print(f"Processed webcam video saved at: {output_path}")
    else:
     print("No webcam frames were saved.")

