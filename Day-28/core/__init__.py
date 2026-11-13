"""Shared computer-vision engine used by both the Gradio and Streamlit apps."""

from .operations import (  # noqa: F401
    CATEGORIES,
    OPERATIONS,
    Operation,
    OpResult,
    Param,
    ProcessingError,
    apply_operation,
    operations_by_category,
)
from .pipeline import PRESETS, Step, format_steps, preset_steps, run_pipeline  # noqa: F401
from .utils import (  # noqa: F401
    EXPORT_FORMATS,
    encode_image,
    ensure_rgb,
    histogram_image,
    image_stats,
    limit_size,
    list_samples,
    load_image,
    save_image,
    wipe_compare,
)

__version__ = "1.0.0"
