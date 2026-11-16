# Dataset sources

20 real images containing text, collected from public sample sets that OCR
projects publish on GitHub. They cover printed pages, books, a newspaper,
a research paper, receipts, an invoice, signboards and handwriting, under
different lighting conditions and text sizes.

| # | File | Category | Lighting / quality | Text size | Source |
|---|---|---|---|---|---|
| 01 | 01_printed_text_12pt.png | Printed document | Clean, digital | Medium (12 pt) | Tesseract test images (`phototest.tif`) |
| 02 | 02_printed_multilingual.png | Printed document | Clean scan | Medium | Tesseract test images (`eurotext.tif`) |
| 03 | 03_typewritten_page_photo.png | Typewritten page | Photo, slightly grey | Medium | OCRmyPDF test resources (`typewriter.png`) |
| 04 | 04_scanned_book_page.png | Book / manual page | Clean scan | Small | OCRmyPDF test resources (`linn.png`) |
| 05 | 05_newspaper_page_small_text.png | Newspaper/magazine page | Black & white scan | Very small, 2 columns | Tesseract test images (`8087_054.3B.tif`) |
| 06 | 06_research_paper_page.png | Research paper | Clean | Small, tables + figures | PaddleOCR docs (`ppstructure/docs/table/1.png`) |
| 07 | 07_table_low_resolution.jpg | Table | Low resolution (371 px) | Tiny | PaddleOCR docs (`ppstructure/docs/table/table.jpg`) |
| 08 | 08_wall_notice_uneven_light.jpg | Notice on a wall | Uneven light, grey | Small | PaddleOCR samples (`doc/imgs_en/img_12.jpg`) |
| 09 | 09_receipt_phone_photo.jpg | Receipt | Phone photo, shadow at bottom | Medium | PaddleOCR samples (`doc/imgs_en/254.jpg`) |
| 10 | 10_receipt_scan_clean.jpg | Receipt | Clean scan | Small | ICDAR 2019 SROIE (`data/img/005.jpg`) |
| 11 | 11_receipt_scan_long.jpg | Receipt | Long, faded thermal print | Small | ICDAR 2019 SROIE (`data/img/000.jpg`) |
| 12 | 12_receipt_low_contrast.jpg | Receipt (tax invoice) | Grey, low contrast photo | Small | ICDAR 2019 SROIE (`data/img/014.jpg`) |
| 13 | 13_tax_invoice_scan.jpg | Invoice | Scan with handwritten mark | Small | ICDAR 2019 SROIE (`data/img/008.jpg`) |
| 14 | 14_signboard_outdoor_large_text.jpg | Signboard | Outdoor daylight, curved logo text | Large | PaddleOCR samples (`doc/imgs_en/img623.jpg`) |
| 15 | 15_park_sign_small_text.jpg | Signboard | Outdoor, sign is small in frame | Small in frame | PaddleOCR samples (`doc/imgs_en/img_11.jpg`) |
| 16 | 16_indoor_sign_low_light.jpg | Signboard | Indoor, low light | Small in frame | PaddleOCR samples (`doc/imgs_en/img_195.jpg`) |
| 17 | 17_large_italic_text.png | Printed text | Clean | Very large, italic | Tesseract test images (`deslant.tif`) |
| 18 | 18_handwritten_page.jpg | Handwriting | Clean scan | Medium | WordDetectorNN test data (CVL dataset sample) |
| 19 | 19_handwritten_line.png | Handwriting | Clean | Medium | SimpleHTR sample data (`line.png`) |
| 20 | 20_historic_handwriting.jpg | Historic handwriting | Old yellowed paper | Small, cursive | WordDetectorNN test data (Bentham collection sample) |

Repositories:
- Tesseract test images: https://github.com/tesseract-ocr/test (Apache 2.0)
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR, branch `release/2.7` (Apache 2.0)
- OCRmyPDF: https://github.com/ocrmypdf/OCRmyPDF, `tests/resources`
- ICDAR 2019 SROIE receipts: https://github.com/zzzDavid/ICDAR-2019-SROIE (research dataset)
- SimpleHTR: https://github.com/githubharald/SimpleHTR (MIT)
- WordDetectorNN: https://github.com/githubharald/WordDetectorNN (MIT)

TIFF files were converted to PNG; no other changes were made to the images.
They are used here for learning and testing only. Check each source before
using the images in anything public or commercial.

**Add your own photos:** drop any `.jpg`, `.png`, `.bmp`, `.tif` or `.webp`
file into this folder and run `python main.py --batch dataset` again.
