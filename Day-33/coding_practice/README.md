# YOLO People Detection and Tracking

A Python command-line application that detects and tracks people using YOLO11 and ByteTrack.

## Features

- Detect people in an image, video file, or webcam
- Track individual people across video frames
- Display confidence scores and tracking IDs
- Show the current frame people count
- Resize the live preview to a readable 1280 x 720 maximum
- Save full-resolution processed output

## Project Structure

```text
coding_practice/
|-- main.py
|-- requirements.txt
|-- yolo11n.pt
|-- input/
|   |-- badminton.mp4
|   |-- group_meeting.mp4
|   `-- track.jpg
`-- output/
    |-- image_detection_output.jpg
    `-- people_detection_output.mp4
```

## Installation

Run these commands from the `Day-33/coding_practice` directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Process a video

```powershell
python main.py --mode video --source input\badminton.mp4
```

### Process an image

```powershell
python main.py --mode image --source input\track.jpg
```

### Use the webcam

```powershell
python main.py --mode video --source 0
```

Press `Q` to stop video or webcam processing.

## Output

- Processed images are saved in `output/image_detection_output.jpg`.
- Processed videos are saved in `output/people_detection_output.mp4`.
- The live preview is resized for readability; the output video is written at the original source dimensions.

## Requirements

- Python 3.10 or 3.11
- OpenCV
- Ultralytics
- A webcam for source `0`, or an image/video file

