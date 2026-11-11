from webcam_utils import render_webcam
import streamlit as st

from video_utils import process_uploaded_video


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Video Processing Studio",
    page_icon="🎥",
    layout="wide"
)


# =====================================================
# HEADER
# =====================================================

st.title("🎥 AI Video Processing Studio")

st.write(
    "A Computer Vision application built with Python, "
    "OpenCV, Streamlit and WebRTC."
)

st.caption(
    "Upload videos or use your live webcam "
    "to explore different image-processing techniques."
)

st.divider()


# =====================================================
# APPLICATION TABS
# =====================================================

video_tab, webcam_tab = st.tabs([
    "🎬 Video Upload",
    "📷 Live Webcam"
])


# =====================================================
# TAB 1: VIDEO UPLOAD
# =====================================================

with video_tab:

    st.header("1. Upload Your Video")

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="uploaded_video"
    )

    st.divider()

    # ---------------------------------------------
    # FILTER SELECTION
    # ---------------------------------------------

    st.header("2. Select Processing Techniques")

    selected_filters = st.multiselect(
        "Choose one or more filters",
        options=[
            "Grayscale",
            "Gaussian Blur",
            "Canny Edge Detection",
            "All"
        ],
        default=["Grayscale"],
        key="video_filters"
    )

    # ---------------------------------------------
    # ADVANCED SETTINGS
    # ---------------------------------------------

    with st.expander("⚙️ Advanced Processing Settings"):

        blur_kernel = st.slider(
            "Gaussian Blur Kernel Size",
            min_value=3,
            max_value=15,
            value=5,
            step=2,
            key="video_blur_kernel"
        )

        canny_lower = st.slider(
            "Canny Lower Threshold",
            min_value=0,
            max_value=255,
            value=100,
            key="video_canny_lower"
        )

        canny_upper = st.slider(
            "Canny Upper Threshold",
            min_value=0,
            max_value=255,
            value=200,
            key="video_canny_upper"
        )

    st.divider()

    # ---------------------------------------------
    # PROCESS VIDEO
    # ---------------------------------------------

    if st.button(
        "🚀 Process Video",
        type="primary",
        use_container_width=True,
        key="process_video_button"
    ):

        if uploaded_video is None:

            st.warning("Please upload a video first.")

        elif not selected_filters:

            st.warning(
                "Please select at least one processing technique."
            )

        elif canny_lower >= canny_upper:

            st.warning(
                "Canny upper threshold must be greater "
                "than lower threshold."
            )

        else:

            try:

                with st.spinner(
                    "Processing video... Please wait."
                ):

                    processed_bytes, properties = (
                        process_uploaded_video(
                            uploaded_video,
                            selected_filters,
                            blur_kernel,
                            canny_lower,
                            canny_upper
                        )
                    )

                # Save results in session state

                st.session_state["processed_video"] = (
                    processed_bytes
                )

                st.session_state["video_properties"] = (
                    properties
                )

                st.session_state["original_video"] = (
                    uploaded_video.getvalue()
                )

                st.session_state["processed_filename"] = (
                    uploaded_video.name.rsplit(".", 1)[0]
                    + "_processed.mp4"
                )

                st.session_state["processed_source_name"] = (
                    uploaded_video.name
                )

                st.success(
                    "Video processed successfully!"
                )

            except Exception as error:

                st.error(
                    f"Processing failed: {error}"
                )

    # ---------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------

    if (
        "processed_video" in st.session_state
        and "original_video" in st.session_state
    ):

        st.divider()

        st.header("3. Original vs Processed Video")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Video")

            st.video(
                st.session_state["original_video"]
            )

        with col2:

            st.subheader("Processed Video")

            st.video(
                st.session_state["processed_video"]
            )

        # -----------------------------------------
        # VIDEO PROPERTIES
        # -----------------------------------------

        st.divider()

        st.header("4. Original Video Properties")

        properties = st.session_state[
            "video_properties"
        ]

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "FPS",
            properties["FPS"]
        )

        col2.metric(
            "Width",
            properties["Width"]
        )

        col3.metric(
            "Height",
            properties["Height"]
        )

        col4.metric(
            "Total Frames",
            properties["Total Frames"]
        )

        col5.metric(
            "Duration",
            f'{properties["Duration (seconds)"]} sec'
        )

        # -----------------------------------------
        # DOWNLOAD PROCESSED VIDEO
        # -----------------------------------------

        st.divider()

        st.header("5. Download Processed Video")

        st.download_button(
            label="⬇️ Download Processed Video",
            data=st.session_state["processed_video"],
            file_name=st.session_state[
                "processed_filename"
            ],
            mime="video/mp4",
            use_container_width=True
        )


# =====================================================
# TAB 2: LIVE WEBCAM
# =====================================================

with webcam_tab:

    render_webcam()


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align: center;">

    *AI Video Processing Studio*

    Built with Python | OpenCV | Streamlit | WebRTC

    Computer Vision Internship Project

    </div>
    """,
    unsafe_allow_html=True
)