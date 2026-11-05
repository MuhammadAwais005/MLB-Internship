
import cv2
import numpy as np
import os


# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FOLDER = r"C:\Users\92348\Desktop\MLBench\Day-25\Document_Image_Enhancement\Dataset\input_documents"
OUTPUT_FOLDER = r"C:\Users\92348\Desktop\MLBench\Day-25\Document_Image_Enhancement\Dataset\outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================================
# 1. LOAD AND VALIDATE IMAGE
# ==========================================================

def load_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image: {image_path}")
        return None

    return image


# ==========================================================
# 2. ORDER DOCUMENT CORNER POINTS
# ==========================================================

def order_points(points):

    points = np.array(points, dtype="float32")

    ordered = np.zeros((4, 2), dtype="float32")

    # Top-left: smallest x + y
    sums = points.sum(axis=1)
    ordered[0] = points[np.argmin(sums)]

    # Bottom-right: largest x + y
    ordered[2] = points[np.argmax(sums)]

    # Top-right: smallest y - x
    differences = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(differences)]

    # Bottom-left: largest y - x
    ordered[3] = points[np.argmax(differences)]

    return ordered


# ==========================================================
# 3. AUTOMATIC PERSPECTIVE CORRECTION
# ==========================================================

def correct_perspective(image):

    original = image.copy()

    height, width = image.shape[:2]

    # Convert temporary image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, 75, 200)

    # Connect nearby edge segments
    kernel = np.ones((3, 3), np.uint8)

    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours by area
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    document_contour = None

    # Search for a four-corner document
    for contour in contours[:20]:

        perimeter = cv2.arcLength(contour, True)

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        area = cv2.contourArea(contour)

        # Document should occupy a reasonable image area
        if (
            len(approximation) == 4
            and cv2.isContourConvex(approximation)
            and area > 0.15 * height * width
        ):

            document_contour = approximation
            break

    # If no document boundary is detected
    if document_contour is None:

        print("  Perspective correction skipped: document boundary not detected.")

        return original, False

    # Arrange four corners
    source_points = order_points(
        document_contour.reshape(4, 2)
    )

    # Calculate output width
    width_a = np.linalg.norm(
        source_points[2] - source_points[3]
    )

    width_b = np.linalg.norm(
        source_points[1] - source_points[0]
    )

    max_width = int(max(width_a, width_b))

    # Calculate output height
    height_a = np.linalg.norm(
        source_points[1] - source_points[2]
    )

    height_b = np.linalg.norm(
        source_points[0] - source_points[3]
    )

    max_height = int(max(height_a, height_b))

    if max_width < 2 or max_height < 2:

        print("  Invalid document dimensions. Skipping correction.")

        return original, False

    # Destination rectangle
    destination_points = np.float32([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ])

    # Calculate perspective matrix
    matrix = cv2.getPerspectiveTransform(
        source_points,
        destination_points
    )

    # Apply perspective transformation
    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    print("  Perspective correction applied successfully.")

    return warped, True


# ==========================================================
# 4. GRAYSCALE CONVERSION
# ==========================================================

def convert_to_grayscale(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    print("  Grayscale conversion completed.")

    return gray


# ==========================================================
# 5. NOISE REDUCTION
# ==========================================================

def reduce_noise(image):

    denoised = cv2.fastNlMeansDenoising(
        image,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    print("  Noise reduction completed.")

    return denoised


# ==========================================================
# 6. BRIGHTNESS AND CONTRAST ENHANCEMENT
# ==========================================================

def enhance_brightness_contrast(image):

    # CLAHE: Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)

    print("  Brightness and contrast enhancement completed.")

    return enhanced


# ==========================================================
# 7. IMAGE SHARPENING
# ==========================================================

def sharpen_image(image):

    # Create a slightly blurred version
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        2
    )

    # Unsharp masking
    sharpened = cv2.addWeighted(
        image,
        1.5,
        blurred,
        -0.5,
        0
    )

    print("  Image sharpening completed.")

    return sharpened


# ==========================================================
# 8. SAVE ENHANCED IMAGE
# ==========================================================

def save_image(image, filename):

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    success = cv2.imwrite(output_path, image)

    if success:

        print(f"  Saved: {output_path}")

    else:

        print(f"  Failed to save: {output_path}")


# ==========================================================
# 9. COMPLETE DOCUMENT ENHANCEMENT PIPELINE
# ==========================================================

def enhance_document(image_path):

    filename = os.path.basename(image_path)

    print("\n" + "=" * 55)
    print(f"Processing: {filename}")
    print("=" * 55)

    # Step 1: Load image
    image = load_image(image_path)

    if image is None:
        return

    print("  Image loaded successfully.")

    # Step 2: Perspective correction
    corrected, perspective_applied = correct_perspective(image)

    # Step 3: Grayscale conversion
    gray = convert_to_grayscale(corrected)

    # Step 4: Noise reduction
    denoised = reduce_noise(gray)

    # Step 5: Brightness and contrast enhancement
    enhanced = enhance_brightness_contrast(denoised)

    # Step 6: Sharpening
    sharpened = sharpen_image(enhanced)

    # Step 7: Save final image
    name_without_extension = os.path.splitext(filename)[0]

    output_filename = f"{name_without_extension}_enhanced.png"

    save_image(sharpened, output_filename)

    print("  Processing completed.")

    return {
        "filename": filename,
        "perspective_corrected": perspective_applied,
        "output": output_filename
    }


# ==========================================================
# 10. PROCESS ENTIRE DATASET
# ==========================================================

def process_dataset():

    if not os.path.exists(INPUT_FOLDER):

        print("Input folder does not exist!")
        return

    supported_formats = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    image_files = sorted([
        file for file in os.listdir(INPUT_FOLDER)
        if file.lower().endswith(supported_formats)
    ])

    if not image_files:

        print("No supported images found in the dataset folder.")
        return

    print("\n==============================================")
    print("       DOCUMENT IMAGE ENHANCEMENT TOOL")
    print("==============================================")

    print(f"Total images found: {len(image_files)}")

    successful = 0
    results = []

    for filename in image_files:

        image_path = os.path.join(
            INPUT_FOLDER,
            filename
        )

        try:

            result = enhance_document(image_path)

            if result is not None:

                successful += 1
                results.append(result)

        except Exception as error:

            print(f"Error processing {filename}: {error}")

    # Final summary
    print("\n==============================================")
    print("              FINAL SUMMARY")
    print("==============================================")

    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {len(image_files) - successful}")

    print(f"\nEnhanced images saved inside: {OUTPUT_FOLDER}")


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    process_dataset()
