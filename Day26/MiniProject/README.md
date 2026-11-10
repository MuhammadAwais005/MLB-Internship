# Mini Project — Document Boundary Detection Tool

A beginner-friendly Python/OpenCV application that loads a document image,
converts it to grayscale, applies Gaussian blur and Canny, cleans its edge map,
selects a large boundary, draws it on the original image and saves the result.

## Run the project

Use Python 3.10 or later. Extract the ZIP, open a terminal inside
`Document_Boundary_Detection`, then run:

```bash
python -m pip install -r requirements.txt
python document_detector.py
```

The default command processes the 15 images in `dataset/images`. Open
`results/<image-name>/document_boundary.png` for each final output. All results
are supplied in the ZIP so you can inspect them before running the code.

Process your own photo:

```bash
python document_detector.py --image "my_document.jpg" --output my_results
```

Process your own folder:

```bash
python document_detector.py --input-dir "my_documents" --output my_results
```

Use a different output directory to preserve an earlier experiment. Reusing an
output directory overwrites matching result filenames. No webcam, API key,
model download or GPU is needed. After dependency installation, processing the
bundled images works offline.

## Required processing steps

| Requirement | Implementation |
|---|---|
| Load a document image | `load_image()` decodes the image as BGR color |
| Convert to grayscale | `cv.cvtColor(..., cv.COLOR_BGR2GRAY)` |
| Apply Gaussian blur | `cv.GaussianBlur(gray, (5, 5), 0)` |
| Detect Canny edges | `cv.Canny(..., low, high, L2gradient=True)` |
| Apply morphology and remove noise | Closing reconnects short gaps; component filtering removes isolated specks |
| Find largest document contour | Sort contours by `cv.contourArea`, descending |
| Draw boundary on original | Scale the selected contour back and draw a green closed outline |
| Save final output | `document_boundary.png` at the original input resolution |

The default selects the **largest plausible convex quadrilateral**. It rejects
tiny blobs and almost-full-image contours. This reduces some background and text
mistakes, but it does not prove that the selected rectangle is a document.
It compares the requested Canny thresholds with a second pair at 75% of those
values, keeping the larger accepted boundary. Use `--single-pass` to disable
that comparison. The intermediate images correspond to the selected attempt.

For the exact basic "largest contour only" version from the assignment, use:

```bash
python document_detector.py --selection largest --output results_largest
```

That mode selects the largest contour above the minimum area and draws its
actual outline, even if it has more than four corners. Compare it with the
default to see how a background shape can affect the simple assumption.

## Dataset: 15 labelled images

| Group | Count | Provenance |
|---|---:|---|
| Source camera photos | 8 | Public examples from Daniel Gatis's `docscan` repository |
| Straight archival scans / reproductions | 3 | Public-domain historical documents via Wikimedia Commons, resized with a visible dark margin |
| Added shadow / uneven illumination | 2 | Brightness-field variants of source photos |
| Slight blur | 2 | Gaussian-blurred source photos |

The source photos already include perspective, tilted documents, different
backgrounds and uneven lighting. There are **11 source images and four additional
variants**, rather than 15 independent captures. The straight images are the Bill
of Rights, the US Constitution (page 1), and Roosevelt's reply to Einstein. They
are existing archival scans/digital reproductions, with added margins for an
observable outer edge. No claim is made that these files come from DocLayNet or Kaggle.

`dataset/manifest.json` records each image's source, modification, dimensions and
SHA-256. `dataset/SOURCES.md`, the retained MIT license and Wikimedia source
metadata provide attribution and source licensing information.
The public identity-document examples already contain blacked-out fields; use
non-sensitive documents for your own testing.

The dataset is bundled. Recreating it is optional and needs internet:

```bash
python make_dataset.py
```

## How the algorithm works

1. Load the original color image and resize a working copy to at most 1000 pixels
   on its longest side. The saved outline remains at the original resolution.
2. Convert the working image to grayscale and blur it with a 5×5 Gaussian kernel.
3. Use Canny thresholds 40 and 120 to obtain a binary edge map. In the default
   mode, also try 30 and 90 and keep the larger accepted boundary.
4. Close the edge map with a rectangular 5×5 kernel. Remove connected components
   smaller than 25 pixels. Closing repairs short breaks; component filtering
   removes small isolated clusters. Opening is deliberately avoided here because
   erosion can erase thin Canny outlines.
5. Extract contours, sort by enclosed area and apply the selected boundary rule.
6. Draw the selected contour in green on a copy of the original and save it.
7. Write intermediate images, status metadata and a batch CSV for review.

If no acceptable boundary exists, the final image is saved without an outline
and `status.json` says `no_boundary`. A plausible contour has status
`boundary_candidate`, which is a proposal to inspect, not measured correctness.

## Tune a difficult image

```bash
python document_detector.py --image "faint.jpg" --low 20 --high 60 --kernel 7
python document_detector.py --image "small_page.jpg" --min-area 0.06
```

- Lower Canny thresholds can recover weak boundaries and introduce extra texture.
- A larger closing kernel can reconnect gaps and also merge unwanted edges.
- A smaller minimum area allows smaller documents and more background candidates.
- A larger `--max-size`, such as 1400, retains detail and takes more processing.

These parameters describe image scale and gradients; there is no universal best
setting. Inspect intermediate images before deciding which setting helps.

## Output files

Each image has its own folder containing:

- `document_boundary.png`: required final full-resolution output.
- `01_original_resized.png`: working-size input.
- `02_gray.png`: grayscale.
- `03_gaussian_blur.png`: smoothed grayscale.
- `04_canny_edges.png`: initial binary edges.
- `05_morphological_closing.png`: gap repair.
- `06_cleaned_edges.png`: edge map after component filtering.
- `status.json`: status, contour area/rank, coordinates, parameters and timing.
  It also records both threshold attempts and the chosen pair.

`results/summary.csv` summarizes the batch. `reports/RESULTS.md` discusses the
actual run and limitations. `reports/all_results.jpg` shows all final overlays.
The walkthrough notebook explains the code and displays examples.

## Limitations

This is a classical contour-based baseline, with no training stage. It assumes
a mostly visible, sufficiently large document outline against a distinguishable
background. Heavy shadows, occlusion, curled pages, weak contrast, nested
rectangles and multiple pages can cause misses or false selections. A tightly
cropped scan with no visible outer edge may produce no boundary; the app does
not silently assume that the entire image is the document. This project does
not perform OCR or perspective correction.

## References

- [OpenCV: contours](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
- [OpenCV: contour area and polygon approximation](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
- [OpenCV: Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [OpenCV: morphology](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [Dataset examples and license](https://github.com/danielgatis/docscan)
- [Bill of Rights archival reproduction](https://commons.wikimedia.org/wiki/File:United_States_Bill_of_Rights.jpg)
- [Constitution archival reproduction](https://commons.wikimedia.org/wiki/File:Constitution_of_the_United_States,_page_1.jpg)
- [Roosevelt letter scan](https://commons.wikimedia.org/wiki/File:Roosevelt-einstein-letter.png)
