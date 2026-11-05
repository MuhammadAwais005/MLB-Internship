# Document Image Enhancement Tool

## Project Overview

The **Document Image Enhancement Tool** is a Python-based application developed using OpenCV and NumPy. It improves the readability and visual quality of document images by applying perspective correction, grayscale conversion, noise reduction, brightness and contrast enhancement, and image sharpening.

The application processes multiple document images automatically and saves the enhanced results in a separate output directory.

## Implemented Techniques

### 1. Perspective Transformation

**Purpose:** To correct perspective distortion when a document is photographed from an angle.

**Implementation:** Used Canny edge detection and contour detection to identify a four-corner document boundary. Then applied `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` to transform the document into a rectangular view.

If a suitable document boundary cannot be detected, the application skips perspective correction and continues with the original image.

### 2. Grayscale Conversion

**Purpose:** To convert the document image from color to grayscale, simplifying subsequent image processing and emphasizing text and background differences.

**Implementation:** Used `cv2.cvtColor()` with `cv2.COLOR_BGR2GRAY`.

### 3. Noise Reduction

**Purpose:** To reduce unwanted visual noise while preserving important text details.

**Implementation:** Used OpenCV's `cv2.fastNlMeansDenoising()` to remove noise from grayscale document images.

### 4. Brightness and Contrast Enhancement

**Purpose:** To improve document readability, particularly when text appears faded or lighting is uneven.

**Implementation:** Applied Contrast Limited Adaptive Histogram Equalization (CLAHE) using `cv2.createCLAHE()`. This enhances local contrast while limiting excessive amplification.

### 5. Image Sharpening

**Purpose:** To improve the visibility of text edges and make document details appear clearer.

**Implementation:** Used Gaussian Blur and weighted image blending with `cv2.addWeighted()` to perform unsharp masking.

## Impact on Document Quality

Perspective correction had the biggest impact on documents photographed at an angle because it helped restore their rectangular shape and improve alignment.

For documents that were already straight, CLAHE-based contrast enhancement and noise reduction were particularly useful for improving readability.

The overall improvement depended on the original image quality and the type of distortion present.

## Challenges Faced

During implementation, several challenges were encountered:

- **Document boundary detection:** Identifying the correct four document corners using edge and contour detection was challenging, especially when the background contained similar rectangular shapes.
- **Perspective correction:** Incorrect corner detection or point ordering could produce distorted output images.
- **Noise reduction:** Excessive denoising could remove fine text details along with unwanted noise.
- **Brightness and contrast:** Enhancing faded text without losing details in brighter areas required careful adjustment.
- **Image sharpening:** Excessive sharpening could introduce unwanted artifacts or make noise more visible.
- **Batch processing:** Handling multiple image files required proper file validation, exception handling, and organized output storage.

## Technologies Used

- Python
- OpenCV
- NumPy

## Project Features

- Automatic loading of multiple document images.
- Automatic document boundary detection.
- Perspective correction when a suitable boundary is detected.
- Grayscale conversion.
- Noise reduction.
- Brightness and contrast enhancement.
- Image sharpening.
- Automatic saving of enhanced images.

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Place document images inside:

```text
dataset/input_documents/
```

Run the application:

```bash
python main.py
```

Enhanced images will be saved in:

```text
outputs/
```

## Conclusion

This project demonstrates how fundamental computer vision and image processing techniques can be combined to improve document image quality. It provides practical experience with geometric transformations, contour detection, filtering, contrast enhancement, and image sharpening using OpenCV.