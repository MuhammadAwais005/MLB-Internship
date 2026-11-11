import cv2


def process_frame(
    frame,
    selected_filters,
    blur_kernel=5,
    canny_lower=100,
    canny_upper=200
):
    """
    Process one BGR video frame using selected filters.

    Available filters:
    - Grayscale
    - Gaussian Blur
    - Canny Edge Detection
    - All
    """

    processed = frame.copy()

    if "All" in selected_filters:
        selected_filters = [
            "Grayscale",
            "Gaussian Blur",
            "Canny Edge Detection"
        ]

    # Grayscale
    if "Grayscale" in selected_filters:
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(
                processed,
                cv2.COLOR_BGR2GRAY
            )

    # Gaussian Blur
    if "Gaussian Blur" in selected_filters:
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(
                processed,
                cv2.COLOR_BGR2GRAY
            )

        processed = cv2.GaussianBlur(
            processed,
            (blur_kernel, blur_kernel),
            0
        )

    # Canny Edge Detection
    if "Canny Edge Detection" in selected_filters:

        if len(processed.shape) == 3:
            processed = cv2.cvtColor(
                processed,
                cv2.COLOR_BGR2GRAY
            )

        processed = cv2.Canny(
            processed,
            canny_lower,
            canny_upper
        )

    return processed


def get_video_properties(cap):

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps if fps > 0 else 0

    return {
        "FPS": round(fps, 2),
        "Width": width,
        "Height": height,
        "Total Frames": total_frames,
        "Duration (seconds)": round(duration, 2)
    }