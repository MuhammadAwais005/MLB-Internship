# Day 29: OCR with EasyOCR

Optical Character Recognition (OCR): reading text from images with **EasyOCR**,
improving the results with image preprocessing, and building a complete
document reader application.

**🌐 Live app:** _link added after deployment_

| Folder | Task | What's inside |
|---|---|---|
| [`EasyOCR/`](EasyOCR) | Coding practice | EasyOCR setup, text extraction from 12 images (printed document, receipt, invoice, signboards, book page, handwritten notes), comparison of results, and grayscale + enhancement preprocessing measured against ground truth |
| [`Mini_Project/`](Mini_Project) | Mini project: **Simple OCR Document Reader** | Web app (Streamlit), desktop app (Tkinter) and command-line batch mode that extract text, show it next to the image and save it to `.txt`, tested on a dataset of 20 real images |

## Highlights

**Coding practice (EasyOCR)**
- 4 preprocessing methods compared on 12 images: original, grayscale, enhanced, binarized.
- Enhancement (shadow removal + denoise + CLAHE + upscaling) raised average word accuracy
  from 92.8% to 95.0% and never made an image worse. Binarization often hurt handwriting.
- Grayscale alone changes almost nothing, because EasyOCR converts to grayscale internally.

**Mini project (OCR Document Reader)**
- Accepts an image (upload, camera, sample or file), extracts all visible text, displays it
  next to the original image with the detected text boxes, and saves/downloads it as `.txt`.
- Rebuilds reading order (lines, paragraphs, columns), so receipts read row by row.
- Optional enhancement for shadowed, faded or tiny text. On a 371 px table it recovered
  almost every missing number.
- Dataset of 20 real public images: books, newspaper, research paper, receipts, invoice,
  signboards and handwriting, under different lighting conditions and text sizes.

## Run locally (Python 3.11.9)

```bat
:: Coding practice
cd Day-29\EasyOCR
py -3.11 -m pip install -r requirements.txt
py -3.11 easyocr_task.py

:: Mini project - web app
cd Day-29\Mini_Project
py -3.11 -m pip install -r requirements.txt
py -3.11 -m streamlit run streamlit_app.py

:: Mini project - desktop app / command line
py -3.11 main.py
py -3.11 main.py --batch dataset
```

Each folder has its own README with full results and explanations.
