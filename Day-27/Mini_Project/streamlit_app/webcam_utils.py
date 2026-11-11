import cv2
import av
import threading
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode
)

from processing_utils import process_frame


class WebcamProcessor(VideoProcessorBase):

    def _init_(self):
        self.selected_filters = ["Grayscale"]
        self.blur_kernel = 5
        self.canny_lower = 100
        self.canny_upper = 200

        self.lock = threading.Lock()

    def update_settings(
        self,
        selected_filters,
        blur_kernel,
        canny_lower,
        canny_upper
    ):
        with self.lock:
            self.selected_filters = selected_filters
            self.blur_kernel = blur_kernel
            self.canny_lower = canny_lower
            self.canny_upper = canny_upper

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        with self.lock:
            filters = list(self.selected_filters)
            kernel = self.blur_kernel
            lower = self.canny_lower
            upper = self.canny_upper

        processed = process_frame(
            image,
            filters,
            kernel,
            lower,
            upper
        )

        # WebRTC expects a three-channel frame
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(
                processed,
                cv2.COLOR_GRAY2BGR
            )

        return av.VideoFrame.from_ndarray(
            processed,
            format="bgr24"
        )


def render_webcam():

    st.header("Live Webcam Processing")

    st.write(
        "Start your webcam and apply computer vision "
        "filters to the live video stream."
    )

    filters = st.multiselect(
        "Select webcam processing techniques",
        options=[
            "Grayscale",
            "Gaussian Blur",
            "Canny Edge Detection",
            "All"
        ],
        default=["Grayscale"],
        key="webcam_filters"
    )

    col1, col2 = st.columns(2)

    with col1:
        kernel = st.slider(
            "Webcam Blur Kernel",
            3, 15, 5, 2,
            key="webcam_kernel"
        )

    with col2:
        lower = st.slider(
            "Webcam Canny Lower Threshold",
            0, 255, 100,
            key="webcam_lower"
        )

    upper = st.slider(
        "Webcam Canny Upper Threshold",
        0, 255, 200,
        key="webcam_upper"
    )

    if lower >= upper:
        st.warning(
            "Canny upper threshold must be greater "
            "than lower threshold."
        )
        return

    def processor_factory():
        processor = WebcamProcessor()

        processor.update_settings(
            filters,
            kernel,
            lower,
            upper
        )

        return processor

    webrtc_streamer(
        key="webcam-processing",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=processor_factory,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )