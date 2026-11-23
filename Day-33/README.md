# Day 33 - YOLO People Detection and Tracking

This project applies YOLO11 and ByteTrack to detect, track, count, and analyze people in images and videos.

## Projects

### Coding Practice

The [coding practice application](coding_practice/README.md) demonstrates command-line person detection and tracking.

- Processes webcam feeds, videos, or images
- Detects only the person COCO class
- Draws bounding boxes, IDs, confidence scores, and people counts
- Saves processed results to `coding_practice/output/`

### Mini Project

The [Streamlit mini project](mini_project/README.md) provides a browser-based interface for image and video analysis.

- Image person detection
- Video person tracking
- Live confidence control
- Peak occupancy analytics
- Downloadable processed images and videos

## Common Setup

Python 3.10 or 3.11 is recommended.

```powershell
cd Day-33
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r coding_practice\requirements.txt
pip install -r mini_project\requirements.txt
```

The Ultralytics YOLO package downloads `yolo11n.pt` when it is not already available locally.

## Run the Coding Practice

```powershell
cd coding_practice
python main.py --mode video --source input\badminton.mp4
python main.py --mode image --source input\track.jpg
python main.py --mode video --source 0
```

Press `Q` to stop live or video processing.

## Run the Mini Project

```powershell
cd mini_project
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

