"""
make_test_images.py
-------------------
Creates the test images used by easyocr_task.py.

Two of the images (the Paris street signboard and the WHO poster) are real
photos taken from the EasyOCR repository. The rest are generated here, and
on purpose they are NOT clean: shadows, faded ink, blur, noise, tilt and so
on, because that is what you get when you take a photo of a document with a
phone. Those problems are exactly what the preprocessing step should fix.

Because we create the images ourselves, we also know the exact text that is
on them. That text is saved in images/ground_truth/ and lets the main script
measure accuracy instead of just guessing "looks better".

Run once:
    python make_test_images.py

Tip: replace or add your own phone photos in the images/ folder any time.
     A ground truth file is optional - without it you just get the text,
     confidence and word count (no accuracy number).
"""

import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
TRUTH_DIR = os.path.join(IMAGE_DIR, "ground_truth")
FONT_DIR = os.path.join(BASE_DIR, "fonts")

# Same seed = same images every time you run it
random.seed(7)
np.random.seed(7)


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


# ---------------------------------------------------------------------------
# Small helpers to make a clean image look like a real-world photo
# ---------------------------------------------------------------------------

def paper_background(width, height, color=(245, 243, 236), grain=6):
    """Off-white paper with a little grain, not a perfect flat white."""
    paper = np.full((height, width, 3), color, dtype=np.float32)
    paper += np.random.normal(0, grain, (height, width, 1))
    return Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8))


def add_shadow(img, strength=0.55, from_left=True):
    """Dark gradient across the page, like your hand or phone blocking light."""
    h, w = img.shape[:2]
    ramp = np.linspace(1 - strength, 1.0, w) if from_left else np.linspace(1.0, 1 - strength, w)
    mask = np.tile(ramp, (h, 1))[..., None]
    return np.clip(img * mask, 0, 255).astype(np.uint8)


def add_noise(img, amount=12):
    noise = np.random.normal(0, amount, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def reduce_contrast(img, factor=0.5, shift=60):
    """Pulls everything towards gray - faded ink / bad lighting."""
    return np.clip(img.astype(np.float32) * factor + shift, 0, 255).astype(np.uint8)


def rotate(img, angle, border=(60, 60, 60)):
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderValue=border)


def perspective(img, shift=0.06, border=(55, 50, 45)):
    """Slight trapezoid, as if the photo was taken at an angle."""
    h, w = img.shape[:2]
    dx, dy = int(w * shift), int(h * shift * 0.5)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[dx, dy], [w - dx // 3, 0], [w, h], [0, h - dy]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h), borderValue=border)


def jpeg_quality(img, quality=45):
    """Re-encode as a low quality JPEG, like WhatsApp does to photos."""
    ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def draw_lines(draw, x, y, lines, fnt, color=(25, 25, 25), spacing=1.45):
    """Draws a list of lines and returns the y position after the last one."""
    line_height = int(fnt.size * spacing)
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=color)
        y += line_height
    return y


def save(name, pil_or_array, truth_lines, bgr_effects=None):
    """Converts to OpenCV, applies effects, saves the image and its ground truth."""
    if isinstance(pil_or_array, Image.Image):
        img = cv2.cvtColor(np.array(pil_or_array), cv2.COLOR_RGB2BGR)
    else:
        img = pil_or_array

    for effect in bgr_effects or []:
        img = effect(img)

    cv2.imwrite(os.path.join(IMAGE_DIR, name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    truth_file = os.path.join(TRUTH_DIR, os.path.splitext(name)[0] + ".txt")
    with open(truth_file, "w", encoding="utf-8") as f:
        f.write("\n".join(truth_lines))

    print(f"  created {name}")


# ---------------------------------------------------------------------------
# The test images
# ---------------------------------------------------------------------------

def printed_document():
    """01 - A normal printed letter, scanned. The 'easy' baseline."""
    lines = [
        "TechNova Solutions (Pvt) Ltd",
        "45-B Main Boulevard, Gulberg III, Lahore",
        "",
        "Date: 12 November 2026",
        "Subject: Internship Offer Letter",
        "",
        "Dear Candidate,",
        "We are pleased to offer you the position of",
        "Machine Learning Intern for a period of three months.",
        "Your internship will start on 1 December 2026.",
        "Please bring your CNIC and degree certificate",
        "on the first day of joining.",
        "",
        "Regards,",
        "HR Department",
    ]
    page = paper_background(1240, 1100, color=(250, 250, 248), grain=3)
    draw = ImageDraw.Draw(page)
    draw.text((90, 70), lines[0], font=font("LiberationSans-Bold.ttf", 46), fill=(20, 40, 90))
    draw_lines(draw, 90, 135, lines[1:], font("LiberationSerif-Regular.ttf", 34))

    save("01_printed_document.jpg", page, [l for l in lines if l],
         [lambda im: rotate(im, -1.2, border=(250, 250, 248)), lambda im: add_noise(im, 4)])


def printed_document_shadow():
    """02 - Same kind of document, but a bad phone photo: shadow + angle + low contrast."""
    lines = [
        "NOTICE FOR ALL STAFF",
        "The office will remain closed on Friday",
        "due to scheduled electrical maintenance.",
        "All servers will be shut down at 6 PM.",
        "Please save your work and log out",
        "before leaving on Thursday evening.",
        "For any queries contact the IT desk",
        "at extension 214.",
    ]
    page = paper_background(1200, 820, grain=5)
    draw = ImageDraw.Draw(page)
    draw.text((80, 60), lines[0], font=font("LiberationSans-Bold.ttf", 52), fill=(30, 30, 30))
    draw_lines(draw, 80, 160, lines[1:], font("LiberationSans-Regular.ttf", 38), color=(55, 55, 55))

    save("02_printed_document_shadow.jpg", page, lines, [
        lambda im: reduce_contrast(im, 0.55, 55),
        lambda im: add_shadow(im, 0.7, from_left=True),
        lambda im: perspective(im, 0.07),
        lambda im: cv2.GaussianBlur(im, (3, 3), 0),
        lambda im: add_noise(im, 8),
        lambda im: jpeg_quality(im, 50),
    ])


def receipt():
    """03 - Thermal receipt: faded gray print, small mono font, on a dark table."""
    lines = [
        "AL-FATAH SUPER STORE",
        "Mall Road, Lahore",
        "Tel: 042-3571-8890",
        "--------------------------------",
        "Milk 1L          x2      560.00",
        "Bread Large      x1      220.00",
        "Eggs Dozen       x1      390.00",
        "Basmati Rice 5kg x1     1850.00",
        "Tea 950g         x1     1499.00",
        "--------------------------------",
        "SUBTOTAL                4519.00",
        "GST 18%                  813.42",
        "TOTAL                   5332.42",
        "CASH                    6000.00",
        "CHANGE                   667.58",
        "--------------------------------",
        "Thank you for shopping!",
    ]
    paper = paper_background(620, 1080, color=(238, 236, 230), grain=4)
    draw = ImageDraw.Draw(paper)
    draw_lines(draw, 32, 40, lines, font("LiberationMono-Regular.ttf", 26), color=(150, 148, 145), spacing=1.65)

    # put the receipt on a dark wooden-ish table
    table = np.full((1250, 900, 3), (40, 55, 75), dtype=np.uint8)
    table = add_noise(table, 10)
    receipt_bgr = cv2.cvtColor(np.array(paper), cv2.COLOR_RGB2BGR)
    table[85:85 + 1080, 140:140 + 620] = receipt_bgr

    truth = [l for l in lines if not l.startswith("---")]
    save("03_receipt.jpg", table, truth, [
        lambda im: rotate(im, 3.5, border=(40, 55, 75)),
        lambda im: add_shadow(im, 0.35, from_left=False),
        lambda im: cv2.GaussianBlur(im, (3, 3), 0),
        lambda im: jpeg_quality(im, 55),
    ])


def invoice():
    """04 - Invoice with a table. Tests reading structured text."""
    page = paper_background(1240, 1000, color=(252, 252, 252), grain=3)
    draw = ImageDraw.Draw(page)
    bold, normal = font("LiberationSans-Bold.ttf", 30), font("LiberationSans-Regular.ttf", 28)

    draw.text((80, 60), "INVOICE", font=font("LiberationSans-Bold.ttf", 64), fill=(180, 30, 30))
    draw_lines(draw, 80, 160, ["Invoice No: INV-2026-0418", "Date: 05 November 2026",
                               "Bill To: Pixel Craft Studio, Karachi"], normal)
    truth = ["INVOICE", "Invoice No: INV-2026-0418", "Date: 05 November 2026",
             "Bill To: Pixel Craft Studio, Karachi"]

    rows = [("Description", "Qty", "Rate", "Amount"),
            ("Website Design", "1", "45000", "45000"),
            ("Logo Design", "2", "8000", "16000"),
            ("Hosting (1 year)", "1", "12000", "12000"),
            ("SEO Setup", "1", "15000", "15000")]
    columns = [80, 640, 780, 960]
    top = 330
    for i, row in enumerate(rows):
        y = top + i * 62
        draw.rectangle([70, y - 10, 1160, y + 52], outline=(120, 120, 120), width=2)
        for x, cell in zip(columns, row):
            draw.text((x, y), cell, font=bold if i == 0 else normal, fill=(30, 30, 30))
        truth.append(" ".join(row))

    totals = ["Subtotal: 88000", "Tax (16%): 14080", "Total Due: PKR 102080"]
    draw_lines(draw, 760, top + len(rows) * 62 + 40, totals, bold)
    truth += totals

    save("04_invoice.jpg", page, truth, [
        lambda im: rotate(im, 0.8, border=(252, 252, 252)),
        lambda im: add_noise(im, 5),
        lambda im: jpeg_quality(im, 70),
    ])


def book_page():
    """07 - Book page: serif font, yellowish old paper, darker towards the spine."""
    lines = [
        "CHAPTER THREE",
        "",
        "The old library stood at the end of the street,",
        "its windows covered in a thin layer of dust.",
        "Nobody remembered when it had last been opened,",
        "yet every evening a single lamp burned inside.",
        "Ahmed had walked past it a thousand times, but",
        "tonight the heavy wooden door was slightly open.",
        "He hesitated for a moment, then stepped inside,",
        "and the smell of ancient paper filled the air.",
        "",
        "47",
    ]
    page = paper_background(1100, 1300, color=(236, 224, 196), grain=7)
    draw = ImageDraw.Draw(page)
    draw.text((330, 90), lines[0], font=font("LiberationSerif-Bold.ttf", 44), fill=(40, 30, 25))
    draw_lines(draw, 90, 230, lines[2:10], font("LiberationSerif-Regular.ttf", 34), color=(45, 35, 30), spacing=1.7)
    draw.text((530, 1180), "47", font=font("LiberationSerif-Regular.ttf", 30), fill=(45, 35, 30))

    save("07_book_page.jpg", page, [l for l in lines if l], [
        lambda im: add_shadow(im, 0.6, from_left=True),   # spine side is darker
        lambda im: perspective(im, 0.04),
        lambda im: add_noise(im, 7),
        lambda im: jpeg_quality(im, 60),
    ])


def handwritten_note():
    """08 - Handwritten-style note in blue ink on lined paper."""
    lines = [
        "Meeting with Sir Usman",
        "Tuesday at 11 am",
        "Bring the project report",
        "and the dataset on USB",
        "Call Bilal about the API key",
    ]
    page = paper_background(1100, 800, color=(250, 250, 244), grain=4)
    draw = ImageDraw.Draw(page)
    for y in range(110, 800, 110):                      # ruled lines
        draw.line([(0, y), (1100, y)], fill=(170, 200, 230), width=2)
    draw.line([(90, 0), (90, 800)], fill=(230, 150, 150), width=2)  # margin line

    hand = font("Caveat.ttf", 72)
    for i, line in enumerate(lines):
        x = 120 + random.randint(-8, 8)
        draw.text((x, 40 + i * 110), line, font=hand, fill=(25, 45, 140))

    save("08_handwritten_note.jpg", page, lines, [
        lambda im: rotate(im, -2.0, border=(70, 70, 70)),
        lambda im: add_noise(im, 6),
        lambda im: jpeg_quality(im, 65),
    ])


def handwritten_list():
    """09 - Pencil shopping list: light gray strokes, low contrast. Hard case."""
    lines = [
        "Shopping list",
        "Tomatoes 1 kg",
        "Onions 2 kg",
        "Yogurt",
        "Green chillies",
        "Cooking oil 3 litre",
    ]
    page = paper_background(900, 900, color=(242, 240, 232), grain=6)
    draw = ImageDraw.Draw(page)
    draw.text((90, 50), lines[0], font=font("Kalam-Regular.ttf", 64), fill=(105, 105, 105))
    hand = font("PatrickHand-Regular.ttf", 56)
    for i, line in enumerate(lines[1:]):
        draw.text((110 + random.randint(-10, 10), 180 + i * 120), "- " + line, font=hand, fill=(120, 120, 118))

    save("09_handwritten_list.jpg", page, lines, [
        lambda im: reduce_contrast(im, 0.75, 40),
        lambda im: add_shadow(im, 0.4, from_left=False),
        lambda im: cv2.GaussianBlur(im, (3, 3), 0),
        lambda im: add_noise(im, 9),
        lambda im: jpeg_quality(im, 50),
    ])


def shop_signboard():
    """10 - Shop signboard on a brick wall, evening light."""
    wall = np.zeros((700, 1200, 3), dtype=np.uint8)
    wall[:] = (60, 75, 130)                                  # brick red (BGR)
    for y in range(0, 700, 40):                              # brick rows
        cv2.line(wall, (0, y), (1200, y), (150, 160, 170), 3)
        offset = 0 if (y // 40) % 2 == 0 else 60
        for x in range(offset, 1200, 120):
            cv2.line(wall, (x, y), (x, y + 40), (150, 160, 170), 3)
    wall = add_noise(wall, 14)

    board = Image.new("RGB", (900, 360), (20, 90, 40))       # green board
    draw = ImageDraw.Draw(board)
    draw.rectangle([10, 10, 889, 349], outline=(240, 200, 60), width=6)
    draw.text((80, 40), "KHAN MOBILE CENTER", font=font("Oswald.ttf", 82), fill=(255, 230, 90))
    draw.text((150, 175), "Repairing - Accessories - Easypaisa", font=font("LiberationSans-Bold.ttf", 34), fill=(255, 255, 255))
    draw.text((270, 250), "0300-1234567", font=font("LiberationSans-Bold.ttf", 50), fill=(255, 255, 255))

    wall[170:530, 150:1050] = cv2.cvtColor(np.array(board), cv2.COLOR_RGB2BGR)

    truth = ["KHAN MOBILE CENTER", "Repairing - Accessories - Easypaisa", "0300-1234567"]
    save("10_shop_signboard.jpg", wall, truth, [
        lambda im: perspective(im, 0.08, border=(30, 30, 30)),
        lambda im: (im * 0.35).astype(np.uint8),             # evening = dark
        lambda im: add_noise(im, 10),
        lambda im: cv2.GaussianBlur(im, (5, 5), 0),
        lambda im: jpeg_quality(im, 45),
    ])


def blurry_label():
    """11 - Small, out of focus product label. Tests upscaling + sharpening."""
    lines = ["ORGANIC GREEN TEA", "Net Wt. 250g", "Best Before: 03/2028", "Made in Pakistan"]
    label = Image.new("RGB", (800, 420), (255, 252, 240))
    draw = ImageDraw.Draw(label)
    draw.rectangle([8, 8, 791, 411], outline=(40, 120, 60), width=8)
    draw.text((70, 50), lines[0], font=font("LiberationSans-Bold.ttf", 60), fill=(30, 100, 50))
    draw_lines(draw, 70, 170, lines[1:], font("LiberationSans-Regular.ttf", 44), color=(40, 40, 40))

    img = cv2.cvtColor(np.array(label), cv2.COLOR_RGB2BGR)
    img = cv2.GaussianBlur(img, (11, 11), 0)                 # out of focus
    img = cv2.resize(img, (230, 121), interpolation=cv2.INTER_AREA)  # tiny photo

    save("11_blurry_label.jpg", img, lines, [lambda im: add_noise(im, 5), lambda im: jpeg_quality(im, 60)])


def faded_photocopy():
    """12 - Faded photocopy with speckle noise, the classic office nightmare."""
    lines = [
        "UNIVERSITY OF THE PUNJAB",
        "Department of Computer Science",
        "Final Term Examination Schedule",
        "Data Structures - 10 January",
        "Operating Systems - 13 January",
        "Artificial Intelligence - 16 January",
        "Computer Networks - 19 January",
    ]
    page = paper_background(1150, 800, color=(235, 235, 235), grain=10)
    draw = ImageDraw.Draw(page)
    draw.text((140, 50), lines[0], font=font("LiberationSerif-Bold.ttf", 52), fill=(80, 80, 80))
    draw_lines(draw, 100, 160, lines[1:], font("LiberationSerif-Regular.ttf", 40), color=(110, 110, 110), spacing=1.6)

    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    # salt and pepper speckles from a dirty photocopier glass
    speckles = np.random.rand(*img.shape[:2])
    img[speckles < 0.012] = 40
    img[speckles > 0.993] = 255

    save("12_faded_photocopy.jpg", img, lines, [
        lambda im: reduce_contrast(im, 0.6, 85),
        lambda im: rotate(im, 1.5, border=(200, 200, 200)),
        lambda im: jpeg_quality(im, 55),
    ])


def write_real_photo_truth():
    """Ground truth for the two real photos (typed by hand from the images)."""
    real = {
        "05_signboard_street": [
            "Mairie du Ier", "Palais du LOUVRE", "LES ARTS DECORATIFS",
            "Musee du LOUVRE", "Theatre du PALAIS-ROYAL",
        ],
        "06_poster_notice": [
            "Reduce your risk of coronavirus infection:",
            "Clean hands with soap and water or alcohol-based hand rub",
            "Cover nose and mouth when coughing and sneezing with tissue or flexed elbow",
            "Avoid close contact with anyone with cold or flu-like symptoms",
            "Thoroughly cook meat and eggs",
            "No unprotected contact with live wild or farm animals",
            "World Health Organization",
        ],
    }
    for name, lines in real.items():
        with open(os.path.join(TRUTH_DIR, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    os.makedirs(TRUTH_DIR, exist_ok=True)
    print("Generating test images...")

    printed_document()
    printed_document_shadow()
    receipt()
    invoice()
    book_page()
    handwritten_note()
    handwritten_list()
    shop_signboard()
    blurry_label()
    faded_photocopy()
    write_real_photo_truth()

    print(f"Done. Images are in: {IMAGE_DIR}")
