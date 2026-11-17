"""
Document & Object Segmentation Tool - command line
==================================================

Segments the main object or document from the background with seven
thresholding techniques, shows how each one did, and saves the best result.

Usage
-----
    python main.py --image photo.jpg              one image
    python main.py --batch dataset                every image in a folder
    python main.py --image photo.jpg --method otsu        save Otsu instead of the automatic pick
    python main.py --batch dataset --save-all     also save every method's mask and cut-out
    python main.py --image coins.png --keep-all   keep every object, not just the main one(s)

Web app:  streamlit run streamlit_app.py

Outputs per image (outputs/<image name>/):
    <name>_comparison.jpg          every method side by side, best in green
    <name>_best_cutout.png         best foreground, transparent background
    <name>_best_mask.png           best mask (white = foreground)
    <name>_best_overlay.jpg        background dimmed, object outlined
    <name>_document_scan.jpg       straightened page (documents only)
    <name>_document_scan_bw.png    black & white scan (documents only)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from segmenter import METHODS, segment
from segmenter.config import IMAGE_TYPES, OUTPUT_DIR
from segmenter.io_utils import load_image, save_image
from segmenter.output import _caption, _tile, comparison_sheet, overlay, save_all_methods, save_best, write_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Document & Object Segmentation Tool")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, help="segment one image")
    mode.add_argument("--batch", type=Path, metavar="FOLDER", help="segment every image in a folder")
    parser.add_argument("--method", choices=list(METHODS), help="save this method instead of the automatic best")
    parser.add_argument("--keep-all", action="store_true", help="keep every object, not only the main one(s)")
    parser.add_argument("--save-all", action="store_true", help="also save each method's mask and cut-out")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="output folder (default: outputs/)")
    return parser.parse_args()


def process(path, args):
    name = path.stem
    folder = args.output / name
    image = load_image(path)

    start = time.perf_counter()
    seg = segment(image, keep_all=args.keep_all)
    seconds = time.perf_counter() - start

    save_image(folder / f"{name}_comparison.jpg", comparison_sheet(name, seg))
    chosen_key = args.method or seg.best_key
    saved = save_best(seg, folder, name, key=chosen_key)
    if args.save_all:
        save_all_methods(seg, folder)

    print(f"\n{path.name}  ({image.shape[1]}x{image.shape[0]}, {seconds:.1f}s)")
    print(f"  {'rank':<5}{'method':<28}{'score':>6}{'edge':>7}{'contrast':>9}{'solid':>7}{'border':>8}{'fg %':>7}{'objects':>9}")
    rows = []
    for rank, (key, r) in enumerate(seg.results.items(), start=1):
        s = r.score
        marker = " <- best" if key == seg.best_key else ""
        marker += " (saved)" if key == chosen_key and args.method else ""
        print(f"  {rank:<5}{r.label:<28}{s.total:>6.1f}{s.edge:>7.2f}{s.contrast:>9.2f}{s.solidity:>7.2f}"
              f"{s.border:>8.2f}{s.coverage:>7.1f}{r.objects:>9}{marker}")
        rows.append({"image": path.name, "rank": rank, "method": key, "label": r.label,
                     "score": round(s.total, 1), "edge": round(s.edge, 3), "contrast": round(s.contrast, 3),
                     "solidity": round(s.solidity, 3), "border": round(s.border, 3),
                     "foreground_%": round(s.coverage, 1), "objects": r.objects,
                     "time_ms": round(r.seconds * 1000, 1), "best": key == seg.best_key,
                     "document_found": bool(seg.document is not None) if key == seg.best_key else ""})
    doc = "  document page found -> straightened scan saved" if "scan" in saved else ""
    print(f"  saved: {seg.results[chosen_key].label}{doc}")

    best = seg.results[chosen_key]
    tile = _caption(_tile(overlay(image, best.mask, seg.document if chosen_key == seg.best_key else None), 300),
                    [name[:34], f"{best.label}  score {best.score.total:.0f}"], True)
    return rows, tile


def main():
    args = parse_args()
    if args.image:
        if not args.image.is_file():
            sys.exit(f"Image not found: {args.image}")
        paths = [args.image]
    else:
        if not args.batch.is_dir():
            sys.exit(f"Folder not found: {args.batch}")
        paths = sorted(p for p in args.batch.iterdir() if p.suffix.lower() in IMAGE_TYPES)
        if not paths:
            sys.exit(f"No images in {args.batch}")

    all_rows, tiles = [], []
    started = time.perf_counter()
    for path in paths:
        try:
            rows, tile = process(path, args)
        except Exception as error:          # one unreadable file should not stop a batch
            print(f"\n{path.name}: FAILED - {error}")
            continue
        all_rows += rows
        tiles.append(tile)

    if not all_rows:
        sys.exit("Nothing was processed.")

    write_summary(all_rows, args.output / "summary.csv")
    if len(tiles) > 1:
        while len(tiles) % 4:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        save_image(args.output / "best_results_overview.jpg", grid)

    best_counts = {}
    for row in all_rows:
        if row["best"]:
            best_counts[row["label"]] = best_counts.get(row["label"], 0) + 1
    print(f"\nDone: {len(tiles)} image(s) in {time.perf_counter() - started:.0f}s")
    if len(tiles) > 1:
        print("Best method per image:", ", ".join(f"{k} x{v}" for k, v in sorted(best_counts.items(), key=lambda x: -x[1])))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
