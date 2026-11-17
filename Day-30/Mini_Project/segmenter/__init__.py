"""Document & Object Segmentation Tool - thresholding-based foreground extraction."""

from .methods import METHODS
from .pipeline import Segmentation, segment

__all__ = ["METHODS", "Segmentation", "segment"]
__version__ = "1.0.0"
