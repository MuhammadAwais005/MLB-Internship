"""Build a reproducible contact sheet and factual run report after detection."""
from pathlib import Path
import json
import sys

import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def build_review(results_dir=None):
    results_dir = Path(results_dir) if results_dir else ROOT / "results"
    report_dir = results_dir.parent / ("reports" if results_dir.name == "results" else results_dir.name + "_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "dataset" / "manifest.json").read_text())
    records = manifest["images"]
    fig, axes = plt.subplots(5, 3, figsize=(12, 20), facecolor="#f5f7fb")
    rows = []
    scan_scores = []
    for ax, record in zip(axes.flat, records):
        folder = results_dir / record["file"].replace(".", "_")
        status_path = folder / "status.json"
        if not status_path.exists():
            ax.text(0.5, 0.5, "No saved result", ha="center", va="center")
            ax.set_title(record["file"], fontsize=10)
            ax.axis("off")
            rows.append((record["file"], "missing", "—", "—"))
            continue
        info = json.loads(status_path.read_text())
        image = cv.imread(str(folder / "document_boundary.png"))
        ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        label = "candidate" if info["status"] == "boundary_candidate" else "no boundary"
        ax.set_title(f"{record['file']} | {label}", fontsize=10)
        ax.axis("off")
        rows.append((record["file"], info["status"], info["contour_rank"], info["area_ratio"]))
        if record["kind"] == "archival_scan":
            h, w = image.shape[:2]
            reference = np.zeros((h, w), np.uint8)
            reference[45:h-45, 45:w-45] = 1
            prediction = np.zeros_like(reference)
            if info["boundary_points"] is not None:
                cv.fillPoly(prediction, [np.array(info["boundary_points"], np.int32)], 1)
            intersection = np.count_nonzero(reference & prediction)
            union = np.count_nonzero(reference | prediction)
            scan_scores.append((record["file"], intersection / union if union else 0.0))
    fig.suptitle("Document boundary detection | 15-image review", fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012, "Green outlines are algorithm proposals. Inspect them for correctness.", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.95, bottom=0.045, hspace=0.25, wspace=0.10)
    fig.savefig(report_dir / "all_results.jpg", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    candidates = sum(row[1] == "boundary_candidate" for row in rows)
    report = [
        "# Document Boundary Detection — Run Report", "",
        f"Processed **{len(records)} test images**. Proposed a boundary on **{candidates}**.", "",
        "This is a candidate count, not an accuracy score. Large rectangles and shadows can still be selected incorrectly.", "",
        "## Configuration", "",
        "Default run: Gaussian blur 5×5; Canny pairs 40/120 and 30/90; rectangular closing kernel 5×5; isolated edge components below 25 pixels removed; minimum area 12%; longest processing side 1000px. For each pair select the largest plausible convex quadrilateral, then keep the larger proposal. Final overlay is saved at original resolution. Each status.json records the attempts and chosen parameters.", "",
        f"Runtime: Python {sys.version.split()[0]}, OpenCV {cv.__version__}, NumPy {np.__version__}.", "",
        "## Per-image outcome", "",
        "| Image | Outcome | Selected contour rank by area | Area / image |",
        "|---|---|---:|---:|",
    ]
    report.extend(f"| {file} | {status} | {rank} | {area} |" for file, status, rank, area in rows)
    report += ["", "## Limited geometry check on the three bordered archival scans", "",
               "For these three files only, the inserted dark border defines a known source-page canvas rectangle. Mask IoU compares the selected polygon with that rectangle. It does not measure the exact irregular edge of physical archival paper or evaluate the camera-photo dataset.", "",
               "| Image | Mask IoU |", "|---|---:|"]
    report.extend(f"| {file} | {score:.4f} |" for file, score in scan_scores)
    report += ["", "## Interpretation and limits", "",
               "The images cover perspective/tilt, varying backgrounds, added illumination gradients, slight blur and straight archival scans/reproductions. Eleven are source images, of which three archival images have added margins; four further inputs are shadow/blur derivatives of source photos. No training is performed.", "",
               "There are no manual ground-truth boundaries for the camera photos. Review all_results.jpg and each full-size overlay. Do not interpret the candidate count or the constructed-page IoUs as general real-world accuracy.", "",
               "Parameters were inspected and adjusted using this same practice set, including a lower-threshold attempt to recover a faint exterior edge rather than a stronger internal rectangle. These are demonstration results, not held-out evaluation.", "",
               "Heavy shadows, curls, low contrast, occlusion and distracting rectangles remain possible failure cases. A no_boundary result is saved without an invented outline. For the strict largest-contour baseline, run document_detector.py --selection largest --output results_largest.", "",
               "## Reproduce", "", "```bash", "python document_detector.py", "python review_results.py", "python -m unittest discover -s tests -v", "```", "",
               "Dataset provenance and license: dataset/SOURCES.md, dataset/manifest.json and dataset/licenses/ (MIT notice and Wikimedia source metadata).", ""]
    (report_dir / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote contact sheet and report: {report_dir}")
    return report_dir


if __name__ == "__main__":
    build_review()
