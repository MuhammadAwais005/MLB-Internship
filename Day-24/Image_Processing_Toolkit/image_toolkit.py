
import cv2
import numpy as np


class ImageToolkit:

    def __init__(self):
        self.original = None
        self.processed = None

    # Load image
    def load_image(self, path):
        image = cv2.imread(path)

        if image is None:
            print("Error: Could not load image!")
            return False

        self.original = image
        self.processed = image.copy()

        print("Image loaded successfully!")
        print("Dimensions:", image.shape)
        return True

    # Convert to grayscale
    def grayscale(self):
        self._check_image()
        self.processed = cv2.cvtColor(
            self.processed, cv2.COLOR_BGR2GRAY
        )
        print("Converted to grayscale.")

    # Resize
    def resize(self, width, height):
        self._check_image()
        self.processed = cv2.resize(
            self.processed, (width, height)
        )
        print("Image resized successfully.")

    # Rotate
    def rotate(self, angle):
        self._check_image()

        rotations = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }

        self.processed = cv2.rotate(
            self.processed, rotations[angle]
        )
        print(f"Image rotated by {angle} degrees.")

    # Flip
    def flip(self, direction):
        self._check_image()

        codes = {
            "horizontal": 1,
            "vertical": 0,
            "both": -1
        }

        self.processed = cv2.flip(
            self.processed, codes[direction]
        )
        print(f"Image flipped: {direction}")

    # Crop
    def crop(self, x1, y1, x2, y2):
        self._check_image()

        h, w = self.processed.shape[:2]

        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            print("Invalid crop coordinates!")
            return

        self.processed = self.processed[y1:y2, x1:x2].copy()
        print("Image cropped successfully.")

    # Draw shapes
    def draw_shapes(self):
        self._check_image()

        # Ensure color image for colored shapes
        if len(self.processed.shape) == 2:
            self.processed = cv2.cvtColor(
                self.processed, cv2.COLOR_GRAY2BGR
            )

        image = self.processed

        cv2.rectangle(
            image, (50, 50), (250, 200),
            (0, 255, 0), 3
        )

        cv2.circle(
            image, (350, 150), 50,
            (0, 0, 255), 3
        )

        cv2.line(
            image, (50, 250), (450, 250),
            (255, 0, 0), 3
        )

        points = np.array([
            [500, 50],
            [600, 150],
            [550, 250],
            [450, 200]
        ], dtype=np.int32)

        cv2.polylines(
            image, [points], True,
            (255, 255, 0), 3
        )

        print("Shapes drawn successfully.")

    # Add custom text
    def add_text(self, text):
        self._check_image()

        if len(self.processed.shape) == 2:
            self.processed = cv2.cvtColor(
                self.processed, cv2.COLOR_GRAY2BGR
            )

        cv2.putText(
            self.processed,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        print("Text added successfully.")

    # Adjust brightness
    def adjust_brightness(self, value):
        self._check_image()

        self.processed = cv2.convertScaleAbs(
            self.processed, alpha=1, beta=value
        )

        print("Brightness adjusted.")

    # Adjust contrast
    def adjust_contrast(self, value):
        self._check_image()

        self.processed = cv2.convertScaleAbs(
            self.processed, alpha=value, beta=0
        )

        print("Contrast adjusted.")

    # BGR vs RGB comparison
    def compare_color_channels(self):
        self._check_image()

        if len(self.processed.shape) == 2:
            print("Convert the image to color before comparing BGR/RGB.")
            return

        rgb = cv2.cvtColor(
            self.processed, cv2.COLOR_BGR2RGB
        )

        cv2.imshow("Original BGR", self.processed)
        cv2.imshow("RGB data displayed as BGR (swapped colors)", rgb)

        print("Press any key to close the comparison windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Display original and processed side by side
    def compare_images(self):
        self._check_image()

        original = self.original.copy()
        processed = self.processed.copy()

        # Convert grayscale to BGR for comparison
        if len(original.shape) == 2:
            original = cv2.cvtColor(
                original, cv2.COLOR_GRAY2BGR
            )

        if len(processed.shape) == 2:
            processed = cv2.cvtColor(
                processed, cv2.COLOR_GRAY2BGR
            )

        # Match heights while preserving aspect ratio
        target_height = 500

        def resize_height(img):
            h, w = img.shape[:2]
            new_width = int(w * target_height / h)
            return cv2.resize(img, (new_width, target_height))

        original = resize_height(original)
        processed = resize_height(processed)

        comparison = np.hstack((original, processed))

        cv2.imshow("Original | Processed", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save processed image
    def save_image(self, path):
        self._check_image()

        success = cv2.imwrite(path, self.processed)

        if success:
            print("Image saved successfully:", path)
        else:
            print("Error saving image!")

    # Internal validation
    def _check_image(self):
        if self.processed is None:
            raise ValueError("Please load an image first!")
