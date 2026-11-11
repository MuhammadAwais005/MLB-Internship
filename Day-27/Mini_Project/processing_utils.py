
import cv2


def process_frame(frame):
    """
    Apply grayscale conversion, Gaussian Blur,
    and Canny Edge Detection to an input frame.

    Returns:
        gray: Grayscale frame
        blurred: Gaussian blurred frame
        edges: Canny edge detection result
    """

    # Step 1: Convert BGR frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian Blur
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Step 3: Apply Canny Edge Detection
    edges = cv2.Canny(
        blurred,
        100,
        200
    )

    return gray, blurred, edges


def display_frames(original, edges):
    """
    Display original and processed frames side by side at a screen-friendly size.
    """
    height, width = original.shape[:2]
    preview_scale = min(1100 / (2 * width), 700 / height, 1.0)
    preview_size = (
        max(1, int(width * preview_scale)),
        max(1, int(height * preview_scale)),
    )

    original_preview = cv2.resize(
        original, preview_size, interpolation=cv2.INTER_AREA
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
    window_name = "Original and Canny Edge Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, preview)
