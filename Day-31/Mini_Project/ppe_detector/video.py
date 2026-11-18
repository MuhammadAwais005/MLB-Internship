"""
Video processing: read frames, detect, draw, and write a browser-playable MP4.

Two details that matter in practice:
  * Browsers only play H.264 MP4. OpenCV's pip wheels cannot write H.264, so
    frames are piped to the ffmpeg binary that ships with imageio-ffmpeg
    (libx264, yuv420p, faststart) - no system ffmpeg needed on Streamlit Cloud.
  * Running the model on every frame is slow on a CPU. With detect_every=N the
    model runs on every N-th frame and the boxes are re-drawn on the frames in
    between, so the output keeps the original length and frame rate.
"""

import os

import cv2
import imageio_ffmpeg

from .compliance import count_by_class
from .settings import CLASS_NAMES
from .visualize import draw_detections


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open the video. Supported: mp4, mov, avi, mkv, webm.")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["seconds"] = info["frames"] / info["fps"] if info["fps"] else 0
    return info


def process_video(input_path, output_path, detector, conf=0.25, iou=0.45, classes=None,
                  detect_every=1, max_seconds=None, max_width=1280, progress=None):
    """
    Writes the annotated video to output_path and returns
        (frame_rows, summary)
    frame_rows: one dict per processed frame with the count of each class
    summary   : frames written, frames sent to the model, peak counts per class
    progress  : optional callback progress(done_fraction) for a progress bar
    """
    info = video_info(input_path)
    fps = info["fps"]
    limit = info["frames"]
    if max_seconds:
        limit = min(limit, int(max_seconds * fps)) if limit > 0 else int(max_seconds * fps)

    # very large frames are slow to encode and to show; scale them down
    scale = min(1.0, max_width / info["width"]) if info["width"] else 1.0
    size = (int(info["width"] * scale) // 2 * 2, int(info["height"] * scale) // 2 * 2)   # libx264 needs even sizes

    # use the ffmpeg bundled with imageio-ffmpeg (it has libx264); a system ffmpeg
    # picked up through IMAGEIO_FFMPEG_EXE may have been built without it
    os.environ.pop("IMAGEIO_FFMPEG_EXE", None)
    writer = imageio_ffmpeg.write_frames(
        str(output_path), size, fps=fps, codec="libx264", pix_fmt_in="bgr24",
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-preset", "veryfast", "-crf", "23"],
        macro_block_size=1, ffmpeg_log_level="error",
    )
    writer.send(None)   # start ffmpeg

    cap = cv2.VideoCapture(str(input_path))
    frame_rows, peak = [], {name: 0 for name in CLASS_NAMES.values()}
    detections, index, model_runs = [], 0, 0
    try:
        while limit <= 0 or index < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0 or frame.shape[1::-1] != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

            if index % detect_every == 0:
                detections = detector.detect(frame, conf=conf, iou=iou, classes=classes)
                model_runs += 1
                counts = count_by_class(detections)
                frame_rows.append({"frame": index, "time_s": round(index / fps, 2),
                                   **{name: counts.get(name, 0) for name in CLASS_NAMES.values()}})
                for name in peak:
                    peak[name] = max(peak[name], counts.get(name, 0))

            writer.send(draw_detections(frame, detections).tobytes())
            index += 1
            if progress and limit > 0 and index % 5 == 0:
                progress(min(index / limit, 1.0))
    finally:
        cap.release()
        writer.close()

    if progress:
        progress(1.0)
    summary = {"frames_written": index, "frames_detected": model_runs, "fps": round(fps, 2),
               "seconds": round(index / fps, 2), "size": f"{size[0]}x{size[1]}", "peak_counts": peak}
    return frame_rows, summary
