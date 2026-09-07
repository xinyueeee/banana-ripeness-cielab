"""
feature_extraction.py  --  the single feature pipeline used everywhere (training
reproduction, tests, Streamlit app).

V3 feature vector = frozen 20 (V1) + 3 enhanced CIELAB / CIEDE2000 descriptors
------------------------------------------------------------------------------
Frozen 20 (unchanged): 15 global CIELAB statistics + foreground fraction +
4 CIEDE2000 prototype distances. Computed by the FROZEN functions in
`cielab_pipeline.py` (verbatim lift of the assignment notebook).

3 V3 additions (greedy 5-fold-CV selection on train+valid only):
    dark_region_mean_L   mean L* of banana pixels with L* < 40   (CIELAB L* statistic)
    h_std                std of hue angle h_ab = atan2(b*, a*)   (cylindrical CIELAB / LCh)
    dE_margin_mean       mean per-pixel (2nd-nearest - nearest) CIEDE2000 distance
                         to the four class prototypes             (CIEDE2000 similarity)

`v3_features.extract_v3_features` computes all 37 CIELAB/CIEDE2000 candidate
descriptors (for the Feature Analysis panel); the model uses the 3 selected ones.
Nothing here is texture / blob / connected-component / a second colour space.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

import cielab_pipeline as frozen
from config import V3_FEATURES
from v3_features import extract_v3_features, lch_maps_for_display


@dataclass
class FeatureBundle:
    feature_vector: np.ndarray            # (23,), ordered as config.V3_FEATURES
    feature_dict: dict[str, float]        # frozen 20 + all 37 V3 CIELAB/CIEDE2000 descriptors
    rgb_resized: np.ndarray
    median_rgb: np.ndarray
    L: np.ndarray
    a: np.ndarray
    b: np.ndarray
    C: np.ndarray                         # chroma  C*_ab  (LCh)
    h: np.ndarray                         # hue angle h_ab  (LCh, degrees)
    mask: np.ndarray                      # uint8 0/255 refined banana mask
    raw_otsu_mask: np.ndarray
    foreground_fraction: float
    b_otsu_threshold: float
    suspicious_mask: bool


def extract_features(rgb_uint8: np.ndarray, prototypes: pd.DataFrame) -> FeatureBundle | None:
    """Run the full V3 pipeline on one 416x416 RGB image.

    Returns None if the frozen segmentation produces an empty mask (no banana).
    """
    channels = frozen.prepare_cielab_channels(rgb_uint8)          # frozen: median + CLAHE(L*)
    seg = frozen.segment_banana(channels["b8"])                  # frozen: Otsu(b*) + morphology
    L, a, b = frozen.to_true_lab_units(channels["L8"], channels["a8"], channels["b8"])

    lab_stats = frozen.extract_lab_features(L, a, b, seg.final_mask)
    if lab_stats is None:
        return None

    row = pd.DataFrame([lab_stats])
    row = frozen.add_ciede2000_features(row, prototypes)          # frozen: 4 CIEDE2000 distances
    v3 = extract_v3_features(L, a, b, seg.final_mask, prototypes)  # 37 CIELAB/CIEDE2000 descriptors

    feature_dict: dict[str, float] = {
        **{k: float(v) for k, v in row.iloc[0].items() if k in frozen.FROZEN_FEATURES},
        **{k: float(v) for k, v in v3.items()},
    }
    feature_vector = np.array([feature_dict[name] for name in V3_FEATURES], dtype=np.float64)

    C, h = lch_maps_for_display(a, b)
    return FeatureBundle(
        feature_vector=feature_vector,
        feature_dict=feature_dict,
        rgb_resized=cv2.resize(rgb_uint8, frozen.EXPECTED_SIZE_HW, interpolation=cv2.INTER_AREA),
        median_rgb=channels["denoised_rgb"],
        L=L, a=a, b=b, C=C, h=h,
        mask=(seg.final_mask > 0).astype(np.uint8) * 255,
        raw_otsu_mask=(seg.b_mask > 0).astype(np.uint8) * 255,
        foreground_fraction=float(seg.foreground_fraction),
        b_otsu_threshold=float(seg.b_otsu_threshold),
        suspicious_mask=bool(seg.suspicious_mask),
    )
