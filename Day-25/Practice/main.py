
import cv2
import numpy as np
import os


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_PATH = r"C:\Users\92348\Desktop\MLBench\Day-25\images\panda.jpg"
OUTPUT_DIR = r"C:\Users\92348\Desktop\MLBench\Day-25\outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(INPUT_PATH)

if image is None:
    raise FileNotFoundError(
        f"Image not found at: {INPUT_PATH}\n"
        "Please place your image inside the images folder "
        "and name it input.jpg."
    )

height, width = image.shape[:2]

print("\n======================================")
print("       IMAGE PROCESSING TOOLKIT")
print("======================================")
print("Image loaded successfully!")
print(f"Width: {width}px")
print(f"Height: {height}px")
print(f"Channels: {image.shape[2] if len(image.shape) == 3 else 1}")


# =========================================================
# HELPER FUNCTION: SAVE AND DISPLAY IMAGE
# =========================================================

def display_result(title, result, filename):

    if result is None:
        print("Error: No result generated.")
        return

    output_path = os.path.join(OUTPUT_DIR, filename)

    cv2.imwrite(output_path, result)

    print(f"\nImage saved successfully: {output_path}")

    # Display original and processed image
    cv2.imshow("Original Image", image)
    cv2.imshow(title, result)

    print("Press any key to close the preview windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================================================
# 1. TRANSLATION
# =========================================================

def translate_image(img, tx, ty):

    h, w = img.shape[:2]

    # Translation matrix
    matrix = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    translated = cv2.warpAffine(img, matrix, (w, h))

    display_result(
        "Translated Image",
        translated,
        "translated.jpg"
    )


# =========================================================
# 2. ROTATION
# =========================================================

def rotate_image(img, angle):

    h, w = img.shape[:2]

    # Find image center
    center = (w // 2, h // 2)

    # Rotation matrix
    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        img,
        matrix,
        (w, h)
    )

    display_result(
        f"Rotated Image ({angle} degrees)",
        rotated,
        f"rotated_{angle}.jpg"
    )


# =========================================================
# 3. SCALING
# =========================================================

def scale_image(img, scale_factor):

    if scale_factor <= 0:
        print("Scale factor must be greater than zero.")
        return

    interpolation = (
        cv2.INTER_CUBIC
        if scale_factor > 1
        else cv2.INTER_AREA
    )

    scaled = cv2.resize(
        img,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=interpolation
    )

    filename = f"scaled_{scale_factor}.jpg"

    display_result(
        f"Scaled Image ({scale_factor}x)",
        scaled,
        filename
    )


# =========================================================
# 4. AFFINE TRANSFORMATION
# =========================================================

def affine_transform(img):

    h, w = img.shape[:2]

    # Three original points
    source_points = np.float32([
        [0.2 * w, 0.2 * h],
        [0.8 * w, 0.2 * h],
        [0.2 * w, 0.8 * h]
    ])

    # Three destination points
    destination_points = np.float32([
        [0.3 * w, 0.1 * h],
        [0.9 * w, 0.3 * h],
        [0.1 * w, 0.9 * h]
    ])

    matrix = cv2.getAffineTransform(
        source_points,
        destination_points
    )

    transformed = cv2.warpAffine(
        img,
        matrix,
        (w, h)
    )

    display_result(
        "Affine Transformation",
        transformed,
        "affine.jpg"
    )


# =========================================================
# 5. PERSPECTIVE TRANSFORMATION
# =========================================================

def perspective_transform(img):

    print("\n--- Perspective Transformation ---")
    print("Enter the four document corner coordinates.")
    print("Required order:")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")

    try:

        points = []

        for i in range(4):

            x = float(input(f"Enter X coordinate of point {i + 1}: "))
            y = float(input(f"Enter Y coordinate of point {i + 1}: "))

            points.append([x, y])

        source_points = np.float32(points)

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

        if max_width < 1 or max_height < 1:
            print("Invalid document dimensions.")
            return

        # Destination rectangle
        destination_points = np.float32([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ])

        # Perspective matrix
        matrix = cv2.getPerspectiveTransform(
            source_points,
            destination_points
        )

        # Apply transformation
        warped = cv2.warpPerspective(
            img,
            matrix,
            (max_width, max_height)
        )

        display_result(
            "Perspective Transformation",
            warped,
            "perspective.jpg"
        )

    except ValueError:
        print("Invalid input. Please enter numerical coordinates.")

    except cv2.error as error:
        print("OpenCV error:", error)


# =========================================================
# 6. BRIGHTNESS ADJUSTMENT
# =========================================================

def adjust_brightness(img, beta):

    # Use floating-point calculations and clip values to 0-255
    result = np.clip(
        img.astype(np.float32) + beta,
        0,
        255
    ).astype(np.uint8)

    filename = f"brightness_{beta}.jpg"

    display_result(
        f"Brightness ({beta})",
        result,
        filename
    )


# =========================================================
# 7. CONTRAST ADJUSTMENT
# =========================================================

def adjust_contrast(img, alpha):

    # alpha controls contrast
    result = np.clip(
        img.astype(np.float32) * alpha,
        0,
        255
    ).astype(np.uint8)

    filename = f"contrast_{alpha}.jpg"

    display_result(
        f"Contrast ({alpha})",
        result,
        filename
    )


# =========================================================
# 8. IMAGE BLURRING
# =========================================================

def apply_blur(img, choice):

    if choice == 1:

        # Gaussian Blur
        result = cv2.GaussianBlur(
            img,
            (5, 5),
            0
        )

        filename = "gaussian_blur.jpg"
        title = "Gaussian Blur"

    elif choice == 2:

        # Median Blur
        result = cv2.medianBlur(
            img,
            5
        )

        filename = "median_blur.jpg"
        title = "Median Blur"

    elif choice == 3:

        # Bilateral Filter
        result = cv2.bilateralFilter(
            img,
            9,
            75,
            75
        )

        filename = "bilateral_filter.jpg"
        title = "Bilateral Filter"

    else:

        print("Invalid blur option.")
        return

    display_result(
        title,
        result,
        filename
    )


# =========================================================
# 9. IMAGE SHARPENING
# =========================================================

def sharpen_image(img):

    # Sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        img,
        -1,
        kernel
    )

    display_result(
        "Sharpened Image",
        sharpened,
        "sharpened.jpg"
    )


# =========================================================
# MAIN MENU
# =========================================================

def main():

    while True:

        print("\n======================================")
        print("          IMAGE PROCESSING MENU")
        print("======================================")

        print("1. Translate Image")
        print("2. Rotate Image")
        print("3. Scale Image")
        print("4. Affine Transformation")
        print("5. Perspective Transformation")
        print("6. Adjust Brightness")
        print("7. Adjust Contrast")
        print("8. Apply Blur Filters")
        print("9. Sharpen Image")
        print("0. Exit")

        print("======================================")

        choice = input("Enter your choice: ").strip()

        try:

            # 1. Translation
            if choice == "1":

                tx = int(input("Enter horizontal translation (tx): "))
                ty = int(input("Enter vertical translation (ty): "))

                translate_image(image, tx, ty)

            # 2. Rotation
            elif choice == "2":

                angle = float(input("Enter rotation angle: "))

                rotate_image(image, angle)

            # 3. Scaling
            elif choice == "3":

                factor = float(input("Enter scale factor (e.g. 1.5 or 0.5): "))

                scale_image(image, factor)

            # 4. Affine Transformation
            elif choice == "4":

                affine_transform(image)

            # 5. Perspective Transformation
            elif choice == "5":

                perspective_transform(image)

            # 6. Brightness
            elif choice == "6":

                beta = int(input("Enter brightness value (-100 to 100): "))

                adjust_brightness(image, beta)

            # 7. Contrast
            elif choice == "7":

                alpha = float(input("Enter contrast factor (e.g. 1.5 or 0.5): "))

                if alpha < 0:
                    print("Contrast factor cannot be negative.")
                else:
                    adjust_contrast(image, alpha)

            # 8. Blur Filters
            elif choice == "8":

                print("\n1. Gaussian Blur")
                print("2. Median Blur")
                print("3. Bilateral Filter")

                blur_choice = int(input("Select blur filter: "))

                apply_blur(image, blur_choice)

            # 9. Sharpening
            elif choice == "9":

                sharpen_image(image)

            # Exit
            elif choice == "0":

                print("\nThank you for using Image Processing Toolkit!")
                break

            else:

                print("Invalid choice. Please select 0-9.")

        except ValueError:

            print("Invalid input! Please enter the correct numerical value.")

        except cv2.error as error:

            print("OpenCV error:", error)

        except Exception as error:

            print("Unexpected error:", error)


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
