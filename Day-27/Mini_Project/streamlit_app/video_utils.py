import cv2
import os
import tempfile
import subprocess
import imageio_ffmpeg

from processing_utils import process_frame, get_video_properties


def process_uploaded_video(
    uploaded_video,
    selected_filters,
    blur_kernel=5,
    canny_lower=100,
    canny_upper=200
):
    """
    Process an uploaded video and convert the result
    to browser-compatible H.264 MP4 format.
    """

    input_path = None
    intermediate_path = None
    final_path = None

    cap = None
    writer = None

    try:
        # -----------------------------------------
        # 1. Save uploaded video temporarily
        # -----------------------------------------

        suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_input:

            temp_input.write(uploaded_video.getvalue())
            input_path = temp_input.name

        # -----------------------------------------
        # 2. Open video
        # -----------------------------------------

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError("Could not open uploaded video.")

        properties = get_video_properties(cap)

        width = properties["Width"]
        height = properties["Height"]
        fps = properties["FPS"]

        if width <= 0 or height <= 0:
            raise ValueError("Invalid video dimensions.")

        if fps <= 0:
            fps = 25

        # -----------------------------------------
        # 3. Create intermediate video
        # -----------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_output:

            intermediate_path = temp_output.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            intermediate_path,
            fourcc,
            fps,
            (width, height),
            True
        )

        if not writer.isOpened():
            raise ValueError("Could not initialize video writer.")

        # -----------------------------------------
        # 4. Process frames
        # -----------------------------------------

        while True:

            success, frame = cap.read()

            if not success:
                break

            processed_frame = process_frame(
                frame,
                selected_filters,
                blur_kernel,
                canny_lower,
                canny_upper
            )

            # Convert grayscale to BGR for video writing
            if len(processed_frame.shape) == 2:

                processed_frame = cv2.cvtColor(
                    processed_frame,
                    cv2.COLOR_GRAY2BGR
                )

            writer.write(processed_frame)

        cap.release()
        cap = None

        writer.release()
        writer = None

        # -----------------------------------------
        # 5. Convert MP4V to browser-compatible H.264
        # -----------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_final:

            final_path = temp_final.name

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_exe,
            "-y",
            "-i", intermediate_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            final_path
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                "H.264 conversion failed: " + result.stderr[-1500:]
            )

        # -----------------------------------------
        # 6. Read final video bytes
        # -----------------------------------------

        with open(final_path, "rb") as file:
            processed_video_bytes = file.read()

        if not processed_video_bytes:
            raise ValueError("Processed video is empty.")

        return processed_video_bytes, properties

    finally:

        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        # Clean temporary files
        for path in [
            input_path,
            intermediate_path,
            final_path
        ]:

            if path and os.path.exists(path):
                os.remove(path)