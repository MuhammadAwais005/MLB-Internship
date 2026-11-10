"""Day 25: edge detection and morphology. Run: python day25.py

Outputs are saved as PNGs, so this also works without desktop GUI windows.
Use your own image: python day25.py --image my_photo.jpg --foreground dark
"""

from pathlib import Path
import argparse
import json
import math

import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def create_samples(folder):
    """Draw reproducible teaching images; no internet download is needed."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(25)
    # BGR color scene: strong boundaries, subtle boundaries and mild noise.
    base = np.tile(np.linspace(30, 85, 800), (500, 1))
    image = np.repeat(base[:, :, None], 3, axis=2)
    image = np.clip(image + rng.normal(0, 5, image.shape), 0, 255).astype(np.uint8)
    cv.rectangle(image, (55, 65), (270, 270), (60, 210, 245), -1)
    cv.circle(image, (420, 170), 100, (215, 150, 65), -1)
    triangle = np.array([[570, 275], [690, 60], [765, 275]], np.int32)
    cv.fillPoly(image, [triangle], (100, 235, 125))
    cv.rectangle(image, (565, 340), (755, 410), (105, 105, 105), -1)
    cv.putText(image, "OpenCV DAY 25", (55, 425), cv.FONT_HERSHEY_SIMPLEX,
               1.5, (245, 245, 245), 3, cv.LINE_AA)
    save_image(folder / "sample_color.png", image)

    # White foreground on black: small white specks, dark holes and a narrow gap.
    binary = np.zeros((360, 640), np.uint8)
    cv.rectangle(binary, (70, 65), (270, 280), 255, -1)
    cv.circle(binary, (430, 180), 95, 255, -1)
    cv.line(binary, (430, 75), (430, 285), 0, 1)
    for center in [(120, 115), (215, 230), (395, 150), (470, 215)]:
        cv.circle(binary, center, 2, 0, -1)
    for center in [(30, 40), (310, 70), (310, 300), (580, 95), (565, 305)]:
        cv.circle(binary, center, 1, 255, -1)
    save_image(folder / "sample_binary.png", binary)

    # Uneven gray background with small bright and dark features for hat filters.
    gray = np.tile(np.linspace(65, 190, 640), (240, 1)).astype(np.uint8)
    cv.circle(gray, (115, 120), 6, 245, -1)
    cv.circle(gray, (520, 120), 6, 15, -1)
    cv.line(gray, (260, 80), (260, 160), 245, 3)
    cv.line(gray, (380, 80), (380, 160), 15, 3)
    save_image(folder / "sample_illumination.png", gray)
    return folder


def display_strength(values):
    """Absolute, per-image normalization for display only, not an edge decision."""
    values = np.abs(values)
    peak = float(values.max())
    if peak == 0:
        return np.zeros(values.shape, np.uint8)
    return np.round(values / peak * 255).astype(np.uint8)


def edge_detection(image, low=50, high=150, blur_size=5):
    if not 0 <= low < high:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high.")
    if blur_size < 1 or blur_size % 2 == 0:
        raise ValueError("Gaussian kernel size must be a positive odd integer.")
    # 1. Grayscale: one intensity channel instead of three BGR channels.
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 2. Smooth BEFORE all three edge methods for the same input comparison.
    blurred = cv.GaussianBlur(gray, (blur_size, blur_size), 0)
    # 3a. Signed floating point preserves positive AND negative derivatives.
    gx = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    magnitude = cv.magnitude(gx, gy)
    # 3b. Laplacian is a second derivative in both spatial directions.
    laplacian = cv.Laplacian(blurred, cv.CV_64F, ksize=3)
    # 3c. Canny gives a binary edge map using two gradient thresholds.
    canny = cv.Canny(blurred, low, high, apertureSize=3, L2gradient=True)
    return {
        "gray": gray, "blurred": blurred,
        "sobel_x": display_strength(gx), "sobel_y": display_strength(gy),
        "sobel_magnitude": display_strength(magnitude),
        "laplacian": display_strength(laplacian), "canny": canny,
    }


def morphological_operations(binary, kernel_size=5):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Morphology kernel size must be a positive odd integer.")
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    # Each operation receives the SAME original binary input independently.
    return {
        "erosion": cv.erode(binary, kernel, iterations=1),
        "dilation": cv.dilate(binary, kernel, iterations=1),
        "opening": cv.morphologyEx(binary, cv.MORPH_OPEN, kernel),
        "closing": cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel),
        "gradient": cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel),
        "top_hat": cv.morphologyEx(binary, cv.MORPH_TOPHAT, kernel),
        "black_hat": cv.morphologyEx(binary, cv.MORPH_BLACKHAT, kernel),
    }


def comparison_grid(items, path, title, columns=3, note=""):
    rows = math.ceil(len(items) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows),
                             squeeze=False, facecolor="#f7f9fc")
    for ax, (label, image) in zip(axes.flat, items):
        if image.ndim == 3:
            ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=12, pad=10)
        ax.axis("off")
    for ax in axes.flat[len(items):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=18, fontweight="bold")
    if note:
        fig.text(0.5, 0.012, note, ha="center", fontsize=10)
    # Reserve explicit row gaps so titles never overlap the preceding image.
    margin = 0.65 / (3.5 * rows)
    fig.subplots_adjust(left=0.025, right=0.975, bottom=margin,
                        top=1 - margin - 0.045, wspace=0.10, hspace=0.38)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def run(image_path=None, output_dir=None, low=50, high=150,
        kernel_size=5, blur_size=5, foreground="bright"):
    samples = create_samples(ROOT / "assets")
    output = Path(output_dir) if output_dir else ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    path = Path(image_path) if image_path else samples / "sample_color.png"
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}. Check the path and format.")
    edges = edge_detection(image, low, high, blur_size)
    for name, result in edges.items():
        save_image(output / f"{name}.png", result)
    comparison_grid([
        ("Original (color)", image), ("Grayscale", edges["gray"]),
        (f"Gaussian blur ({blur_size} x {blur_size})", edges["blurred"]),
        ("Sobel magnitude (display scaled)", edges["sobel_magnitude"]),
        ("Absolute Laplacian (display scaled)", edges["laplacian"]),
        (f"Canny ({low}, {high})", edges["canny"]),
    ], output / "01_edge_comparison.png", "DAY 25 | Edge detection",
        note="Sobel and Laplacian are independently scaled strengths; Canny is a binary map.")
    comparison_grid([
        ("Sobel X: vertical boundaries", edges["sobel_x"]),
        ("Sobel Y: horizontal boundaries", edges["sobel_y"]),
        ("Combined magnitude", edges["sobel_magnitude"]),
    ], output / "02_sobel_directions.png", "Sobel | Direction matters")
    threshold_items = [("Grayscale", edges["gray"])]
    for lo, hi in [(20, 60), (50, 150), (100, 200)]:
        result = cv.Canny(edges["blurred"], lo, hi, L2gradient=True)
        threshold_items.append((f"Canny {lo}/{hi}", result))
    comparison_grid(threshold_items, output / "03_canny_thresholds.png",
                    "Canny | Compare threshold pairs", columns=2)

    if image_path is None:
        binary = cv.imread(str(samples / "sample_binary.png"), cv.IMREAD_GRAYSCALE)
        threshold = None
    else:
        mode = cv.THRESH_BINARY_INV if foreground == "dark" else cv.THRESH_BINARY
        threshold, binary = cv.threshold(edges["gray"], 0, 255, mode | cv.THRESH_OTSU)
    save_image(output / "binary_input.png", binary)
    morph = morphological_operations(binary, kernel_size)
    for name, result in morph.items():
        save_image(output / f"{name}.png", result)
    items = [("Before: binary input", binary)]
    items.extend((name.replace("_", " ").title(), result) for name, result in morph.items())
    comparison_grid(items, output / "04_morphology_comparison.png",
                    f"Morphology | Same input, {kernel_size} x {kernel_size} kernel", columns=4,
                    note="White = foreground. Each operation starts from the binary input.")
    pairs = []
    for name, result in morph.items():
        pairs.extend([("Before: binary input", binary),
                      ("After: " + name.replace("_", " ").title(), result)])
    comparison_grid(pairs, output / "05_before_after.png",
                    "Morphology | Before and after each operation", columns=2)
    illumination = cv.imread(str(samples / "sample_illumination.png"), cv.IMREAD_GRAYSCALE)
    hat_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    comparison_grid([
        ("Uneven background + small features", illumination),
        ("Top hat: small bright features", cv.morphologyEx(illumination, cv.MORPH_TOPHAT, hat_kernel)),
        ("Black hat: small dark features", cv.morphologyEx(illumination, cv.MORPH_BLACKHAT, hat_kernel)),
    ], output / "06_grayscale_hats.png", "Hat filters | Grayscale example")
    metadata = {
        "opencv": cv.__version__, "numpy": np.__version__,
        "input": path.name, "shape": list(image.shape),
        "gaussian_kernel": blur_size, "canny_thresholds": [low, high],
        "morphology_kernel": kernel_size, "otsu_threshold": threshold,
        "white_pixels_before": int(cv.countNonZero(binary)),
        "white_pixels_after": {k: int(cv.countNonZero(v)) for k, v in morph.items()},
        "note": "White pixel counts describe the output, not accuracy or quality.",
    }
    (output / "run_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Completed. Comparison figures and individual images: {output.resolve()}")
    return edges, binary, morph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Optional input image; otherwise use built-in samples")
    parser.add_argument("--output", help="Output folder (default: outputs next to this script)")
    parser.add_argument("--low", type=float, default=50)
    parser.add_argument("--high", type=float, default=150)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--blur", type=int, default=5)
    parser.add_argument("--foreground", choices=["bright", "dark"], default="bright",
                        help="For your own image: should bright or dark pixels be white foreground?")
    args = parser.parse_args()
    try:
        run(args.image, args.output, args.low, args.high, args.kernel, args.blur, args.foreground)
    except (ValueError, FileNotFoundError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
