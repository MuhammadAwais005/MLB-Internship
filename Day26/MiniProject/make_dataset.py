"""Download eight public photos and three archival scans; add four variants.

Bundled data means this is optional. Recreate it: python make_dataset.py
Internet is required only for this preparation step.
"""
from pathlib import Path
import hashlib
import json
import urllib.request
import urllib.parse

import cv2 as cv
import numpy as np
from document_detector import load_image, save_image

ROOT = Path(__file__).resolve().parent
BASE = "https://raw.githubusercontent.com/danielgatis/docscan/master/"


def download(relative, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return BASE + relative
    request = urllib.request.Request(BASE + relative, headers={"User-Agent": "Day25-Internship-Project"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read()
    target.write_bytes(data)
    return BASE + relative


def prepare_dataset():
    dataset = ROOT / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True, exist_ok=True)
    download("LICENSE.txt", dataset / "licenses" / "docscan-MIT.txt")
    records = []

    def record(path, condition, kind, source, change="Unmodified source image"):
        image = load_image(path)
        records.append({
            "file": path.name, "condition": condition, "kind": kind,
            "source_url": source, "modification": change,
            "width": image.shape[1], "height": image.shape[0],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    conditions = [
        "Camera photo: tall brochure, perspective and uneven lighting",
        "Camera photo: letter on wood, mild perspective",
        "Camera photo: tilted landscape paper, dark surroundings",
        "Camera photo: curved identity-document spread",
        "Camera photo: identity-document spread, perspective",
        "Camera photo: identity-document spread, uneven lighting",
        "Camera photo: tilted identity document on a light background",
        "Camera photo: strongly tilted identity document on wood",
    ]
    for index in range(1, 9):
        name = f"doc-{index}" + (".png" if index == 2 else ".jpg")
        target = images / name
        url = download("examples/" + name, target)
        record(target, conditions[index - 1], "source_photo", url)
        print("Downloaded", name)

    # Public-domain straight archival scans/digital reproductions. Retain
    # source metadata, and add a visible margin to the supplied page canvas.
    titles = ["File:United States Bill of Rights.jpg",
              "File:Constitution of the United States, page 1.jpg",
              "File:Roosevelt-einstein-letter.png"]
    params = {"action": "query", "format": "json", "prop": "imageinfo",
              "iiprop": "url|extmetadata", "iiurlwidth": "1000", "redirects": "1",
              "titles": "|".join(titles)}
    api_url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(api_url, headers={"User-Agent": "DocumentBoundaryInternship/1.0 (educational samples)"})
    with urllib.request.urlopen(request, timeout=45) as response:
        source_metadata = json.load(response)
    (dataset / "licenses" / "wikimedia-source-metadata.json").write_text(
        json.dumps(source_metadata, indent=2), encoding="utf-8")
    pages = list(source_metadata["query"]["pages"].values())
    for index, match in enumerate(("Bill of Rights", "Constitution of the United States", "Roosevelt-einstein"), 1):
        page = next(page for page in pages if match in page["title"])
        info = page["imageinfo"][0]
        url = info.get("thumburl", info["url"]).split("?")[0]
        suffix = ".png" if ".png" in url.lower() else ".jpg"
        cache = dataset / "source_scans" / f"scan_{index:02}{suffix}"
        cache.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "DocumentBoundaryInternship/1.0 (educational samples)"})
        with urllib.request.urlopen(request, timeout=45) as response:
            cache.write_bytes(response.read())
        image = load_image(cache)
        scale = min(1.0, 1100 / max(image.shape[:2]))
        image = cv.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                          interpolation=cv.INTER_AREA)
        image = cv.copyMakeBorder(image, 45, 45, 45, 45, cv.BORDER_CONSTANT, value=(45, 45, 45))
        path = images / f"scan_{index:02}.png"
        save_image(path, image)
        source_page = "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_"), safe=":()")
        record(path, "Straight archival scan/digital reproduction with visible margins", "archival_scan", source_page,
               "Public-domain archival scan/reproduction; resized to <=1100px; dark 45px border added")
        records[-1]["download_url"] = url
        records[-1]["license"] = info["extmetadata"].get("LicenseShortName", {}).get("value", "See retained source metadata")
        records[-1]["title"] = page["title"]

    for index, direction in [(1, "horizontal"), (2, "diagonal")]:
        source_name = f"doc-{index}" + (".png" if index == 2 else ".jpg")
        image = load_image(images / source_name).astype(np.float32)
        h, w = image.shape[:2]
        x = np.linspace(0, 1, w)[None, :]
        y = np.linspace(0, 1, h)[:, None]
        field = np.broadcast_to(0.38 + 0.62 * x, (h, w)) if direction == "horizontal" else 0.38 + 0.62 * (x + y) / 2
        shadowed = np.clip(image * field[:, :, None], 0, 255).astype(np.uint8)
        path = images / f"shadow_{index:02}.png"
        save_image(path, shadowed)
        record(path, f"Added {direction} uneven-lighting field", "augmented_photo", BASE + "examples/" + source_name,
               f"Brightness multiplied by a {direction} field ranging from 0.38 to 1.0")

    for index in (3, 8):
        name = f"doc-{index}.jpg"
        image = cv.GaussianBlur(load_image(images / name), (7, 7), 1.4)
        path = images / f"blur_{index:02}.png"
        save_image(path, image)
        record(path, "Slight Gaussian blur on camera photo", "augmented_photo", BASE + "examples/" + name,
               "7x7 Gaussian blur, sigma=1.4")

    manifest = {
        "name": "Document boundary mini-project: 15-image practice set",
        "source_repository": "https://github.com/danielgatis/docscan",
        "credit": "Photos: Daniel Gatis, docscan (MIT). Archival scans: US National Archives / Franklin D. Roosevelt, via Wikimedia Commons (public domain labels retained).",
        "license_file": "licenses/docscan-MIT.txt",
        "composition": {"original_photos": 8, "archival_scans": 3,
                        "added_shadows": 2, "added_blur": 2},
        "limitations": "Four variants share source photos. The archival scans/digital reproductions have added margins. This is not 15 independent scenes, a training set, or a benchmark.",
        "images": records,
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(records)} images in {images}")


if __name__ == "__main__":
    prepare_dataset()
