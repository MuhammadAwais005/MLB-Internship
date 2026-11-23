# Smart People Counting System

A browser-based Streamlit application for detecting and tracking people in images and videos using YOLO11 and ByteTrack.

## Features

- Upload JPG, JPEG, or PNG images
- Upload MP4, AVI, MOV, or MKV videos
- Adjust the detection confidence from the sidebar
- Detect only people (COCO class 0)
- Draw bounding boxes, tracking IDs, and confidence scores
- Display the current people count
- Calculate peak video occupancy
- Download processed images and browser-compatible H.264 videos

## Project Structure

```text
mini_project/
|-- app.py
|-- requirements.txt
|-- yolo11n.pt
|-- sample_videos/
`-- outputs/
```

## Installation

Run these commands from the `Day-33/mini_project` directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

```powershell
streamlit run app.py
```

Open the URL displayed by Streamlit, then use the **Image Detection** or **Video Detection** tab.

## Detection Settings

The **Detection Confidence** slider controls the minimum confidence required for a person to be detected. Its default value is `0.35`.

## How It Works

1. The YOLO11 nano model detects person objects.
2. Bounding boxes and labels are drawn for each person.
3. ByteTrack assigns and maintains IDs for people in videos.
4. The app updates the current count and peak occupancy for each frame.
5. Processed files are saved in `outputs/` and made available for download.

Video files are converted to H.264 when FFmpeg is available so they work correctly in browsers.

## Requirements

- Python 3.10 or 3.11
- Streamlit
- Ultralytics
- OpenCV
- NumPy
- Pillow
- imageio-FFmpeg
