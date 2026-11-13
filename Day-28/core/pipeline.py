"""
Multi-step pipelines ("multiple filters in sequence") and ready-made presets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from .operations import OPERATIONS, ProcessingError, apply_operation


@dataclass
class Step:
    op_key: str
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def op(self):
        return OPERATIONS[self.op_key]

    def describe(self) -> str:
        return self.op.summarize(self.params)

    def to_dict(self) -> Dict[str, Any]:
        return {"op_key": self.op_key, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step":
        return cls(data["op_key"], dict(data.get("params", {})))


@dataclass
class StepReport:
    index: int
    description: str
    info: str
    millis: float
    image: np.ndarray


def run_pipeline(img: np.ndarray, steps: List[Step]) -> Tuple[np.ndarray, List[StepReport]]:
    """Apply steps in order. Returns the final image and a per-step report."""
    if not steps:
        raise ProcessingError("The pipeline is empty. Add at least one step.")
    current = img
    reports: List[StepReport] = []
    for i, step in enumerate(steps, start=1):
        t0 = time.perf_counter()
        result = apply_operation(current, step.op_key, step.params)
        current = result.image
        reports.append(StepReport(i, step.describe(), result.info, (time.perf_counter() - t0) * 1000, current))
    return current, reports


def format_steps(steps: List[Step]) -> str:
    if not steps:
        return "_Pipeline is empty. Pick an operation, tune it, then press **Add to pipeline**._"
    return "\n".join(f"{i}. {s.op.icon} {s.describe()}" for i, s in enumerate(steps, start=1))


# --------------------------------------------------------------------------- #
# Presets: curated recipes that show off chaining
# --------------------------------------------------------------------------- #
PRESETS: Dict[str, List[Step]] = {
    "📄 Document Scanner": [
        Step("grayscale", {"method": "Luminosity"}),
        Step("denoise", {"method": "Median (fast)", "strength": 6}),
        Step("threshold", {"method": "Adaptive Gaussian", "block": 31, "c": 12}),
        Step("denoise", {"method": "Median (fast)", "strength": 3}),
    ],
    "📸 Vintage Photo": [
        Step("color_adjust", {"saturation": 80, "hue": 0, "warmth": 25}),
        Step("sepia", {"intensity": 70}),
        Step("brightness_contrast", {"brightness": 5, "contrast": -10}),
        Step("vignette", {"strength": 70, "radius": 35}),
    ],
    "🌟 Vivid & Sharp": [
        Step("enhance", {"method": "Auto Enhance", "clip_limit": 2.0, "tiles": 8}),
        Step("color_adjust", {"saturation": 125, "hue": 0, "warmth": 5}),
        Step("sharpen", {"method": "Unsharp Mask", "amount": 0.8, "radius": 1.5}),
    ],
    "🖍️ Comic Book": [
        Step("enhance", {"method": "CLAHE (adaptive)", "clip_limit": 2.5, "tiles": 8}),
        Step("cartoon", {"levels": 6, "smoothness": 6, "line_size": 9}),
        Step("color_adjust", {"saturation": 140, "hue": 0, "warmth": 0}),
    ],
    "🔍 Object Outlines": [
        Step("blur", {"method": "Bilateral", "kernel": 9}),
        Step("contours", {"detection": "Otsu (auto)", "mode": "Outer only", "min_area": 500,
                          "color": "Rainbow", "thickness": 3, "fill": True, "numbers": True}),
    ],
    "🌙 Night Vision": [
        Step("grayscale", {"method": "Luminosity"}),
        Step("gamma", {"gamma": 1.8}),
        Step("color_adjust", {"saturation": 100, "hue": 0, "warmth": -100}),
        Step("vignette", {"strength": 85, "radius": 30}),
    ],
}


def preset_steps(name: str) -> List[Step]:
    return [Step(s.op_key, dict(s.params)) for s in PRESETS.get(name, [])]
