# Document Boundary Detection — Run Report

Processed **15 test images**. Proposed a boundary on **12**.

This is a candidate count, not an accuracy score. Large rectangles and shadows can still be selected incorrectly.

## Configuration

Default run: Gaussian blur 5×5; Canny pairs 40/120 and 30/90; rectangular closing kernel 5×5; isolated edge components below 25 pixels removed; minimum area 12%; longest processing side 1000px. For each pair select the largest plausible convex quadrilateral, then keep the larger proposal. Final overlay is saved at original resolution. Each status.json records the attempts and chosen parameters.

Runtime: Python 3.12.14, OpenCV 4.14.0, NumPy 2.3.5.

## Per-image outcome

| Image | Outcome | Selected contour rank by area | Area / image |
|---|---|---:|---:|
| doc-1.jpg | boundary_candidate | 1 | 0.6801 |
| doc-2.png | boundary_candidate | 1 | 0.6782 |
| doc-3.jpg | boundary_candidate | 1 | 0.5082 |
| doc-4.jpg | boundary_candidate | 1 | 0.5932 |
| doc-5.jpg | no_boundary | None | None |
| doc-6.jpg | boundary_candidate | 1 | 0.6215 |
| doc-7.jpg | no_boundary | None | None |
| doc-8.jpg | no_boundary | None | None |
| scan_01.png | boundary_candidate | 1 | 0.8478 |
| scan_02.png | boundary_candidate | 1 | 0.8383 |
| scan_03.png | boundary_candidate | 1 | 0.7548 |
| shadow_01.png | boundary_candidate | 1 | 0.6801 |
| shadow_02.png | boundary_candidate | 1 | 0.6729 |
| blur_03.png | boundary_candidate | 1 | 0.5078 |
| blur_08.png | boundary_candidate | 1 | 0.2303 |

## Limited geometry check on the three bordered archival scans

For these three files only, the inserted dark border defines a known source-page canvas rectangle. Mask IoU compares the selected polygon with that rectangle. It does not measure the exact irregular edge of physical archival paper or evaluate the camera-photo dataset.

| Image | Mask IoU |
|---|---:|
| scan_01.png | 0.9975 |
| scan_02.png | 0.9973 |
| scan_03.png | 1.0000 |

## Interpretation and limits

The images cover perspective/tilt, varying backgrounds, added illumination gradients, slight blur and straight archival scans/reproductions. Eleven are source images, of which three archival images have added margins; four further inputs are shadow/blur derivatives of source photos. No training is performed.

There are no manual ground-truth boundaries for the camera photos. Review all_results.jpg and each full-size overlay. Do not interpret the candidate count or the constructed-page IoUs as general real-world accuracy.

Parameters were inspected and adjusted using this same practice set, including a lower-threshold attempt to recover a faint exterior edge rather than a stronger internal rectangle. These are demonstration results, not held-out evaluation.

Heavy shadows, curls, low contrast, occlusion and distracting rectangles remain possible failure cases. A no_boundary result is saved without an invented outline. For the strict largest-contour baseline, run document_detector.py --selection largest --output results_largest.

## Reproduce

```bash
python document_detector.py
python review_results.py
python -m unittest discover -s tests -v
```

Dataset provenance and license: dataset/SOURCES.md, dataset/manifest.json and dataset/licenses/ (MIT notice and Wikimedia source metadata).
