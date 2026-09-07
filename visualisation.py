"""
visualisation.py  --  build the image-processing panels and feature tables shown
in the Streamlit UI. Pure functions (no Streamlit calls).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from config import FEATURE_GROUPS, FEATURE_HELP, CLASS_COLOUR
from feature_extraction import FeatureBundle
from theme import CLASS_EMOJI


def _u8(gray: np.ndarray) -> np.ndarray:
    return np.clip(gray, 0, 255).astype(np.uint8)


def pipeline_stage_panels(fb: FeatureBundle) -> list[tuple[str, np.ndarray, str | None]]:
    """original -> median -> L* -> a* -> b* -> C*ab -> h_ab -> mask -> extracted."""
    extracted = fb.rgb_resized.copy()
    extracted[fb.mask == 0] = 0
    # hue angle scaled to 0..255 over the 0..360 range, masked for readability
    h_disp = _u8(fb.h / 360.0 * 255.0)
    return [
        ("Original (416 x 416)", fb.rgb_resized, None),
        ("3 x 3 median filtered", fb.median_rgb, None),
        ("L* channel (after CLAHE)", _u8(fb.L * 2.55), "gray"),
        ("a* channel (green <-> red)", _u8(fb.a + 128), "gray"),
        ("b* channel (blue <-> yellow) -- used for Otsu", _u8(fb.b + 128), "gray"),
        ("C*ab (chroma / colourfulness) -- LCh", _u8(fb.C * 2.0), "gray"),
        ("h_ab (hue angle) -- LCh", h_disp, "gray"),
        ("Banana segmentation mask", fb.mask, "gray"),
        ("Extracted banana region", extracted, None),
    ]


def result_badge_html(predicted_class: str) -> str:
    colour = CLASS_COLOUR.get(predicted_class, "#666")
    emoji = CLASS_EMOJI.get(predicted_class, "🍌")
    label = {"unripe": "UNRIPE", "ripe": "RIPE",
             "overripe": "OVERRIPE", "rotten": "ROTTEN"}.get(predicted_class, predicted_class.upper())
    return (f'<div class="result-badge"><span class="badge-inner" '
            f'style="background:linear-gradient(135deg,{colour}dd,{colour});'
            f'box-shadow:0 4px 12px {colour}55;">'
            f'<span class="dot">{emoji}</span><span>{label}</span></span></div>')


def feature_group_tables(fb: FeatureBundle) -> list[tuple[str, str, pd.DataFrame]]:
    tables = []
    for group, names in FEATURE_GROUPS.items():
        rows = [{"feature": n, "value": round(float(fb.feature_dict.get(n, float("nan"))), 4)}
                for n in names]
        tables.append((group, FEATURE_HELP[group], pd.DataFrame(rows)))
    return tables


def processing_summary(fb: FeatureBundle, proc_ms: float, clf_ms: float, n_features: int) -> pd.DataFrame:
    return pd.DataFrame({
        "Property": ["Image size used", "Colour space", "Noise reduction", "Lighting correction",
                     "Segmentation", "Foreground fraction", "Segmentation quality",
                     "Feature count", "Classifier", "Feature processing time", "Classifier time"],
        "Value": ["416 x 416", "CIELAB (true units) + LCh + CIEDE2000", "3 x 3 median filter",
                  "CLAHE on L* (clip 2.0, 8x8)",
                  f"Otsu on b*  (threshold {fb.b_otsu_threshold:.0f})",
                  f"{fb.foreground_fraction * 100:.1f} %",
                  "suspicious -- check mask" if fb.suspicious_mask else "normal",
                  str(n_features), "SVC (RBF, sklearn defaults)",
                  f"{proc_ms:.0f} ms", f"{clf_ms:.1f} ms"],
    }).set_index("Property")
