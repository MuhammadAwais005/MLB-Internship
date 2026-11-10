# Day 25 — Edge Detection & Morphological Operations

Complete Python/OpenCV internship practice. Start with the notebook for the
explanation and cell-by-cell implementation; use the Python script for repeatable
runs. The included images are generated teaching examples, not real photographs.

## Run in three steps

1. Install Python 3.10 or later and extract this ZIP.
2. Open a terminal inside `Day25_Edge_Detection` and install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run:

   ```bash
   python day25.py
   ```

Open the PNG files inside `outputs` to inspect the results. No webcam, internet
download, or OpenCV display window is required after installation.

For the notebook, open `Day25_Explained.ipynb` in VS Code/Jupyter, select your
Python environment and choose **Run All**. If needed, install Jupyter with
`python -m pip install notebook` and start it with `python -m notebook`.
The notebook also works independently in Google Colab: upload only the notebook,
then run all cells. Its first code cell installs dependencies only if missing.

## Use your own image

```bash
python day25.py --image "my_photo.jpg" --output my_results
python day25.py --image "document.jpg" --foreground dark --output document_results
python day25.py --low 30 --high 90 --blur 5 --kernel 3
```

`--foreground dark` is useful for dark writing on light paper: thresholding
inverts it so the writing becomes white foreground. The script uses Otsu's global
threshold for custom images. Uneven lighting or textured photographs may need a
different segmentation method; a binary threshold is not object recognition.
Gaussian and morphology kernel sizes must be positive odd integers. Paths with
spaces need quotes. A missing or unreadable input produces a clear error.

## 1. Edge detection

An edge is a rapid change in image intensity. Edge detection highlights these
transitions, which can correspond to object boundaries, writing, shadows, or
texture. An edge map alone does not identify an object.

The processing order in this task is color → grayscale → Gaussian blur → edge
operator. Grayscale provides one intensity channel. Gaussian smoothing reduces
small fluctuations before derivatives are computed, though excessive blur can
erase useful detail. All three methods receive the same blurred input.

| Method | Calculation | What to inspect |
|---|---|---|
| Sobel | First derivatives in X and Y; combined strength `sqrt(Gx² + Gy²)` | X emphasizes vertical boundaries; Y emphasizes horizontal boundaries |
| Laplacian | Sum of second derivatives in X and Y | Rapid intensity changes; often sensitive to noise |
| Canny | Gradients, nonmaximum suppression, two-threshold hysteresis | Thin binary edge map; thresholds affect retained edges |

Sobel and Laplacian are calculated in signed floating point (`CV_64F`) to retain
both directions of intensity change. Absolute strengths are normalized to 0–255
for viewing. The displayed Laplacian is an absolute response, not a separate
zero-crossing detector. Canny already returns a 0/255 map. Because derivative
panels use their own display scaling, their brightness cannot establish which
method is more accurate.

### Choosing Canny thresholds

The lower threshold rejects weak gradient candidates. The upper threshold seeds
strong edges; intermediate candidates survive when linked to strong edges.
These are gradient thresholds, not grayscale brightness thresholds, and are not
intrinsically restricted to 0–255 even for an 8-bit input.

Try 20/60, 50/150 and 100/200 on the same blurred image, as shown in
`03_canny_thresholds.png`. These are starting experiments, not universal best
settings. If useful contours disappear, lower the thresholds. If texture dominates,
raise them or adjust smoothing. Judge against the boundaries needed for your task.

### Practical applications

Possible uses include locating a document outline before perspective correction,
finding part boundaries during visual inspection, and preparing contours for
shape analysis. Edge detection can support OCR preprocessing and segmentation,
but neither recognizes text nor provides complete segmentation by itself.

## 2. Morphological operations

Morphology uses a small structuring element (kernel) to examine local shape.
Here, white pixels represent foreground. A rectangular 5×5 kernel is the default;
changing its size or shape changes which details survive. Morphology also works
on grayscale images, as demonstrated separately for hat filters.

| Operation | Definition | Typical effect on white foreground |
|---|---|---|
| Erosion | Local minimum | Shrinks regions and can remove tiny specks |
| Dilation | Local maximum | Expands regions and can bridge narrow gaps |
| Opening | Erosion, then dilation | Removes features smaller than the kernel |
| Closing | Dilation, then erosion | Fills sufficiently small holes or gaps |
| Morphological gradient | Dilation − erosion | Highlights region outlines |
| Top hat | Input − opening | Isolates small bright details |
| Black hat | Closing − input | Isolates small dark details |

Each operation is applied independently to the same binary image. Only opening
and closing contain their own prescribed two-step sequence. These operations can
also remove useful details or merge nearby regions, so they do not automatically
improve every image.

## Results and interpretation

The delivered `outputs` were generated by the included program. Re-run it to
reproduce the comparisons or change settings to experiment.

| Figure | Required comparison |
|---|---|
| `01_edge_comparison.png` | Original, grayscale, blur, Sobel, Laplacian, Canny |
| `02_sobel_directions.png` | X, Y and combined Sobel |
| `03_canny_thresholds.png` | Three Canny threshold pairs |
| `04_morphology_comparison.png` | Binary input and all seven operations |
| `05_before_after.png` | Original input beside each morphological result |
| `06_grayscale_hats.png` | Bright and dark feature extraction on a gray background |

On the built-in binary demo, inspect the isolated white dots outside the shapes,
the black pinholes inside them, and the slit in the circle. Opening removes the
tiny exterior dots; closing fills pinholes and bridges the narrow slit. Erosion
reduces the large shapes and dilation expands them. Top hat exposes the small
bright features removed by opening; black hat exposes dark details filled by
closing. The gradient shows boundary bands. The grayscale hat example makes
their bright-versus-dark behavior visible under a varying background.

For edge comparison, inspect the subtle gray rectangle as well as the high
contrast shapes and letters. Canny produces discrete contours; the derivative
maps retain graded response strengths. No ground-truth labels are supplied, so
this exercise makes visual comparisons rather than reporting detection accuracy.
`run_summary.json` records parameters, package versions and white-pixel counts;
counts describe the images, not quality.

## Files and coverage

- `day25.py`: all six coding requirements, reusable functions and command options.
- `Day25_Explained.ipynb`: self-contained, executed notebook with explanations.
- `assets/`: three reproducible sample images.
- `outputs/`: six comparison figures, individual processed PNGs and run metadata.
- `requirements.txt`: runtime dependencies.

## Short conclusion for the task

The exercise compares directional and second-derivative responses with a binary
Canny edge map after Gaussian smoothing. It then demonstrates how kernel-based
shape operations shrink, expand, clean, connect and outline a foreground mask.
Useful results depend on threshold choice, foreground polarity and kernel scale.

## References

Official OpenCV documentation consulted for definitions and API behavior:

- [Image gradients](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html)
- [Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [Morphological operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [Smoothing](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
