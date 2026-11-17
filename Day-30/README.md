# Day 30: Image Thresholding & Segmentation

Separating objects and documents from their background with thresholding (OpenCV).

| Folder | Task | What's inside |
|---|---|---|
| [`Coding_Practice/`](Coding_Practice) | Coding practice | Grayscale, Binary, Adaptive and Otsu thresholding compared on 4 images with different lighting, plus a simple foreground/background segmentation. All outputs saved for comparison |
| [`Mini_Project/`](Mini_Project) | Mini project: **Document & Object Segmentation Tool** | Streamlit web app + command-line tool. 7 thresholding techniques, automatic quality scoring to pick the best result, document straightening, tested on 16 real images |

## Highlights

**Coding practice**
- Adaptive thresholding is the only method that reads an unevenly lit page; Otsu is best
  for objects (it picked T=104 for the coins instead of the guessed 127 and cut the noise blobs from 39 to 17).
- Segmentation with illumination correction separates all 24 coins and extracts the
  shadowed receipt text.

**Mini project**
- Seven thresholding techniques run side by side with identical cleanup, so only the
  thresholding differs.
- A 0–100 quality score (edge alignment, colour contrast, solidity, border contact) picks
  the best result without ground truth; the user can override it.
- Documents are detected and straightened into a flat scan (all 8 documents in the dataset).
- No single method wins everywhere: across 16 images, 7 different methods were the winner at least once.

## Run (Python 3.11.9)

```bat
:: Coding practice
cd Day-30\Coding_Practice
py -3.11 -m pip install -r requirements.txt
py -3.11 thresholding.py

:: Mini project
cd Day-30\Mini_Project
py -3.11 -m pip install -r requirements.txt
py -3.11 -m streamlit run streamlit_app.py      :: web app
py -3.11 main.py --batch dataset                :: command line, all images
```
