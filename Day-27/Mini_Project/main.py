
import os

from video_processor import process_video
from webcam_processor import process_webcam


# Get current project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define input and output paths
INPUT_VIDEO = os.path.join(
    BASE_DIR,
    "input",
    "Tour.mp4"
)

VIDEO_OUTPUT = os.path.join(
    BASE_DIR,
    "output",
    "processed_video.mp4"
)

WEBCAM_OUTPUT = os.path.join(
    BASE_DIR,
    "output",
    "processed_webcam.mp4"
)


def main():

    while True:

        print("\n====================================")
        print("   REAL-TIME VIDEO PROCESSING TOOL")
        print("====================================")

        print("1. Process Recorded Video")
        print("2. Process Live Webcam")
        print("3. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":

            process_video(
                INPUT_VIDEO,
                VIDEO_OUTPUT
            )

        elif choice == "2":

            process_webcam(
                WEBCAM_OUTPUT
            )

        elif choice == "3":

            print("Thank you for using the application!")
            break

        else:

            print("Invalid choice! Please try again.")


if __name__ == "__main__":
    main()
