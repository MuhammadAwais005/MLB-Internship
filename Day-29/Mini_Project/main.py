"""
Simple OCR Document Reader
==========================

Reads all visible text from an image with EasyOCR, shows it next to the
original image and saves it to a .txt file.

Usage
-----
    python main.py                         open the desktop app
    python main.py --open photo.jpg        open the app with an image already loaded
    python main.py --image photo.jpg       command line: print text, save .txt + preview
    python main.py --image photo.jpg --show   ...and open the preview image
    python main.py --batch dataset         process every image in a folder

Options
-------
    --enhance          clean the image first (shadows, faded print, tiny text)
    --lang en ur       EasyOCR language codes (default: en)
    --output outputs   where .txt files, previews and summary.csv go
"""

import argparse
import sys
import time
from pathlib import Path

from ocr_reader.config import DEFAULT_LANGUAGES, OUTPUT_DIR, SUPPORTED_EXTENSIONS

MIN_PYTHON = (3, 9)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple OCR Document Reader (EasyOCR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run without arguments to open the desktop app.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--image", type=Path, help="read one image from the command line")
    mode.add_argument("--batch", type=Path, metavar="FOLDER", help="read every image in a folder")
    mode.add_argument("--open", type=Path, metavar="IMAGE", help="open the desktop app with this image")
    parser.add_argument("--enhance", action="store_true", help="enhance the image before OCR")
    parser.add_argument("--show", action="store_true", help="open the preview after reading (with --image)")
    parser.add_argument("--lang", nargs="+", default=DEFAULT_LANGUAGES, metavar="CODE",
                        help="EasyOCR language codes, e.g. --lang en ur (default: en)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="output folder (default: outputs/)")
    return parser.parse_args()


def find_images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def print_result(result, txt_path, preview_path):
    line = "-" * 70
    print(f"\n{line}\n{result.image_path.name}\n{line}")
    print(result.text or "(no text found)")
    print(line)
    print(f"{result.line_count} lines | {result.word_count} words | "
          f"avg confidence {result.average_confidence * 100:.1f}% | {result.seconds:.1f}s")
    print(f"Text saved to   : {txt_path}")
    print(f"Preview saved to: {preview_path}")


def read_single(engine, image_path, enhance, output_dir, show):
    from ocr_reader.output import save_preview, save_text

    result = engine.read(image_path, enhance=enhance)
    txt_path = save_text(result, output_dir / "text")
    preview_path = save_preview(result, output_dir / "previews")
    print_result(result, txt_path, preview_path)

    if show:
        from PIL import Image
        Image.open(preview_path).show()      # opens in the default image viewer


def read_batch(engine, folder, enhance, output_dir):
    from ocr_reader.output import save_preview, save_text, write_summary

    images = find_images(folder)
    if not images:
        print(f"No images found in {folder}")
        return

    print(f"Found {len(images)} images in {folder}\n")
    results, failed = [], []
    started = time.perf_counter()

    for number, path in enumerate(images, start=1):
        try:
            result = engine.read(path, enhance=enhance)
        except Exception as error:               # one bad file should not stop the batch
            failed.append(path.name)
            print(f"[{number:>2}/{len(images)}] {path.name:<40} FAILED: {error}")
            continue
        save_text(result, output_dir / "text")
        save_preview(result, output_dir / "previews")
        results.append(result)
        print(f"[{number:>2}/{len(images)}] {path.name:<40} {result.line_count:>3} lines "
              f"{result.word_count:>4} words  conf {result.average_confidence * 100:5.1f}%  "
              f"{result.seconds:5.1f}s")

    summary = write_summary(results, output_dir / "summary.csv")
    total = time.perf_counter() - started
    print(f"\nDone: {len(results)} images in {total:.0f}s"
          + (f", {len(failed)} failed ({', '.join(failed)})" if failed else ""))
    print(f"Text files : {output_dir / 'text'}")
    print(f"Previews   : {output_dir / 'previews'}")
    print(f"Summary    : {summary}")


def main():
    if sys.version_info < MIN_PYTHON:
        sys.exit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required (tested on 3.11.9).")

    args = parse_args()

    if args.image is None and args.batch is None:
        from ocr_reader.gui import run_app
        if args.open and not args.open.is_file():
            sys.exit(f"Image not found: {args.open}")
        run_app(start_image=args.open, languages=args.lang)
        return

    if args.image and not args.image.is_file():
        sys.exit(f"Image not found: {args.image}")
    if args.batch and not args.batch.is_dir():
        sys.exit(f"Folder not found: {args.batch}")

    from ocr_reader.engine import OCREngine
    print(f"Loading EasyOCR ({', '.join(args.lang)})... first run downloads the models (~100 MB).")
    engine = OCREngine(languages=args.lang)
    engine.load()

    if args.image:
        read_single(engine, args.image, args.enhance, args.output, args.show)
    else:
        read_batch(engine, args.batch, args.enhance, args.output)


if __name__ == "__main__":
    main()
