
import cv2
import streamlit as st
import numpy as np
import tempfile
import subprocess
import imageio_ffmpeg

from pathlib import Path
from uuid import uuid4
from ultralytics import YOLO


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart People Counting System",
    page_icon="👥",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD YOLO MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = YOLO("yolo11n.pt")

    return model


# ==========================================
# 3. DRAW DETECTIONS
# ==========================================

def draw_detections(frame, boxes, show_tracking=True):

    people_count = len(boxes)

    for box in boxes:

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # Get confidence score
        confidence = float(box.conf[0])

        # Get tracking ID if available
        track_id = None

        if show_tracking and box.id is not None:
            track_id = int(box.id[0])

        # Prepare detection label
        if track_id is not None:

            label = f"ID: {track_id} | {confidence:.2f}"

        else:

            label = f"Person | {confidence:.2f}"

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw label background
        cv2.rectangle(
            frame,
            (x1, max(y1 - 30, 0)),
            (x1 + 190, y1),
            (0, 255, 0),
            -1
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 5, max(y1 - 9, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )

    return frame, people_count


# ==========================================
# 4. IMAGE PROCESSING
# ==========================================

def process_image(image_file, model, confidence):

    # Read uploaded image
    image_bytes = image_file.getvalue()

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        raise ValueError("Unable to read uploaded image.")

    # YOLO person detection
    results = model.predict(
        source=frame,
        classes=[0],
        conf=confidence,
        verbose=False
    )

    result = results[0]

    boxes = result.boxes

    # Draw detections
    processed_frame, people_count = draw_detections(
        frame,
        boxes,
        show_tracking=False
    )

    # Display people count
    cv2.rectangle(
        processed_frame,
        (10, 10),
        (320, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        processed_frame,
        f"People Count: {people_count}",
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Save processed image
    output_path = OUTPUT_DIR / f"processed_image_{uuid4().hex[:8]}.jpg"

    cv2.imwrite(
        str(output_path),
        processed_frame
    )

    return output_path, people_count


# ==========================================
# 5. VIDEO PROCESSING
# ==========================================

def process_video(video_file, model, confidence, progress_bar, status):

    # Save uploaded video temporarily
    suffix = Path(video_file.name).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(video_file.getvalue())

        input_path = Path(temp_file.name)

    # Open uploaded video
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():

        input_path.unlink(missing_ok=True)

        raise ValueError("Unable to open uploaded video.")

    # Read video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        fps = 30.0

    if width <= 0 or height <= 0:

        cap.release()
        input_path.unlink(missing_ok=True)

        raise ValueError("Invalid video dimensions.")

    # Output file names
    unique_id = uuid4().hex[:8]

    raw_output = OUTPUT_DIR / f"raw_{unique_id}.mp4"

    final_output = OUTPUT_DIR / f"processed_video_{unique_id}.mp4"

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(raw_output),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()
        input_path.unlink(missing_ok=True)

        raise ValueError("Unable to initialize video writer.")

    # Analytics variables
    frame_number = 0

    peak_people_count = 0

    # ======================================
    # PROCESS EACH VIDEO FRAME
    # ======================================

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # YOLO + ByteTrack
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=confidence,
                verbose=False
            )

            result = results[0]

            boxes = result.boxes

            # Draw detections and count people
            processed_frame, people_count = draw_detections(
                frame,
                boxes,
                show_tracking=True
            )

            # Update peak occupancy
            peak_people_count = max(
                peak_people_count,
                people_count
            )

            # Display current people count
            cv2.rectangle(
                processed_frame,
                (10, 10),
                (350, 110),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                processed_frame,
                f"People Count: {people_count}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                processed_frame,
                f"Peak Occupancy: {peak_people_count}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # Save processed frame
            writer.write(processed_frame)

            # Update processing progress
            if total_frames > 0:

                progress = min(
                    frame_number / total_frames,
                    1.0
                )

                progress_bar.progress(progress)

            if frame_number % 10 == 0:

                status.write(
                    f"Processing frame {frame_number} "
                    f"of {total_frames if total_frames > 0 else 'unknown'}"
                )

    finally:

        cap.release()
        writer.release()
        input_path.unlink(missing_ok=True)

    # ======================================
    # CONVERT VIDEO TO H.264
    # FOR BROWSER COMPATIBILITY
    # ======================================

    try:

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel", "error",
                "-i", str(raw_output),
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                str(final_output)
            ],
            check=True,
            capture_output=True
        )

        raw_output.unlink(missing_ok=True)

    except Exception as error:

        print("Video conversion warning:", error)

        # Keep original MP4 if conversion is unavailable
        final_output = raw_output

    progress_bar.progress(1.0)

    status.success("Video processing completed!")

    return final_output, peak_people_count, frame_number


# ==========================================
# 6. STREAMLIT USER INTERFACE
# ==========================================

st.title("👥 Smart People Counting System")

st.markdown(
    """
    Upload an image or video to detect and count people using
    YOLO11 and ByteTrack.

    The application also calculates peak occupancy and
    provides downloadable processed results.
    """
)

st.divider()


# ==========================================
# SIDEBAR SETTINGS
# ==========================================

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05
)

st.sidebar.info(
    "The application detects only people (COCO class 0)."
)


# Load model
try:

    model = load_model()

except Exception as error:

    st.error(f"Unable to load YOLO model: {error}")

    st.stop()


# ==========================================
# MAIN APPLICATION TABS
# ==========================================

image_tab, video_tab = st.tabs(
    ["🖼️ Image Detection", "🎥 Video Detection"]
)


# ==========================================
# IMAGE TAB
# ==========================================

with image_tab:

    st.subheader("Upload an Image")

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Original Image",
            width="stretch"
        )

        if st.button(
            "Detect People in Image",
            key="image_process"
        ):

            with st.spinner("Detecting people..."):

                try:

                    output_path, people_count = process_image(
                        uploaded_image,
                        model,
                        confidence
                    )

                    st.success("Image processing completed!")

                    st.divider()

                    st.subheader("Detection Results")

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "People Detected",
                            people_count
                        )

                    with col2:

                        st.metric(
                            "Detection Confidence",
                            f"{confidence:.2f}"
                        )

                    st.image(
                        str(output_path),
                        caption="Processed Image",
                        width="stretch"
                    )

                    with open(output_path, "rb") as file:

                        st.download_button(
                            "Download Processed Image",
                            data=file.read(),
                            file_name="people_detection.jpg",
                            mime="image/jpeg"
                        )

                except Exception as error:

                    st.error(f"Image processing failed: {error}")


# ==========================================
# VIDEO TAB
# ==========================================

with video_tab:

    st.subheader("Upload a Video")

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button(
            "Process Video",
            key="video_process"
        ):

            progress_bar = st.progress(0)

            status = st.empty()

            with st.spinner("Processing video with YOLO..."):

                try:

                    output_path, peak_count, total_frames = process_video(
                        uploaded_video,
                        model,
                        confidence,
                        progress_bar,
                        status
                    )

                    st.success("Video processing completed!")

                    st.divider()

                    st.subheader("Crowd Analytics")

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Peak Occupancy",
                            peak_count
                        )

                    with col2:

                        st.metric(
                            "Total Frames",
                            total_frames
                        )

                    with col3:

                        st.metric(
                            "Maximum Visible People",
                            peak_count
                        )

                    st.divider()

                    st.subheader("Processed Video")

                    st.video(str(output_path))

                    with open(output_path, "rb") as file:

                        st.download_button(
                            "Download Processed Video",
                            data=file.read(),
                            file_name="people_counting_output.mp4",
                            mime="video/mp4"
                        )

                except Exception as error:

                    st.error(f"Video processing failed: {error}")


# ==========================================
# APPLICATION FOOTER
# ==========================================

st.divider()

st.caption(
    "Smart People Counting System | "
    "Powered by YOLO11, ByteTrack, OpenCV and Streamlit"
)
