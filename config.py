"""
config.py  --  all constants, paths and UI copy for the CIELAB prototype.

The CIELAB approach = the frozen V1 pipeline (416x416 -> 3x3 median -> RGB->CIELAB
-> CLAHE on L* -> Otsu on b* -> morphology -> features) plus 3 descriptors that
stay strictly inside the CIELAB / cylindrical-CIELAB (LCh) / CIEDE2000 space:
    dark_region_mean_L   conditional mean of L*            (CIELAB L* statistic)
    h_std                std of hue angle h_ab             (cylindrical CIELAB / LCh)
    dE_margin_mean       mean per-pixel nearest/2nd-nearest CIEDE2000 gap (CIEDE2000)

Classifier: StandardScaler() -> default SVC() -- the exact shared configuration
used by every approach in the group comparison, with no hyperparameter tuned and
no per-approach classification-stage change. No LBP, no blob analysis, no second
colour space, no hand rules.
"""
from __future__ import annotations
from pathlib import Path

# ------------------------------------------------------------------ paths
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
MODEL_PATH = MODELS_DIR / "cielab_model.joblib"
OOD_PATH = MODELS_DIR / "ood_reference.joblib"
PROTOTYPES_PATH = MODELS_DIR / "training_prototypes.csv"
METADATA_PATH = MODELS_DIR / "metadata.json"

# ------------------------------------------------------------------ classes
CLASS_ORDER = ["unripe", "ripe", "overripe", "rotten"]
CLASS_DISPLAY = {"unripe": "UNRIPE", "ripe": "RIPE", "overripe": "OVERRIPE", "rotten": "ROTTEN"}
CLASS_COLOUR = {"unripe": "#3f9d4e", "ripe": "#e8b923", "overripe": "#c07c25", "rotten": "#6d4c41"}

# ------------------------------------------------------------------ frozen pipeline (unchanged since V1)
IMAGE_SIZE = (416, 416)
MEDIAN_KERNEL = 3
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
SUSPICIOUS_FG_MIN = 0.02
SUSPICIOUS_FG_MAX = 0.90

# ------------------------------------------------------------------ image calibration
# Two-part calibration: (1) spatial scaling -- every image resized to a common
# 416x416 canvas so pixel counts / area fractions are comparable; (2) geometric
# rectification -- the banana's principal axis is rotated to horizontal before
# measurement so the bounding box, aspect ratio and extent are pose-independent.
# The dataset has no physical scale reference, so lengths are reported in pixels.
CANVAS_PX = IMAGE_SIZE[0] * IMAGE_SIZE[1]          # 173,056 px total
CALIBRATION_NOTE = ("Calibration has two steps. Spatial scaling: every image is resized to "
                    "416 x 416 (INTER_AREA) so pixel counts and area fractions are comparable "
                    "between images. Geometric rectification: the banana's principal axis "
                    "(from its min-area rectangle) is rotated to horizontal before the "
                    "bounding box and object measurements are taken, so they do not depend on "
                    "how the banana was oriented in the photo. Rectification is applied for "
                    "measurement only and does not change the 23 classifier features. No "
                    "physical (mm) scale reference exists in this dataset, so lengths are in "
                    "pixels.")

# ------------------------------------------------------------------ feature set (V3 = V1 20 + 3)
FROZEN_20 = [
    "L_mean", "L_median", "L_std", "L_p10", "L_p90",
    "a_mean", "a_median", "a_std", "a_p10", "a_p90",
    "b_mean", "b_median", "b_std", "b_p10", "b_p90",
    "foreground_fraction",
    "ciede2000_to_unripe", "ciede2000_to_ripe", "ciede2000_to_overripe", "ciede2000_to_rotten",
]
V3_EXTRA = ["dark_region_mean_L", "h_std", "dE_margin_mean"]
V3_FEATURES = FROZEN_20 + V3_EXTRA

FEATURE_GROUPS: dict[str, list[str]] = {
    "Global Lab statistics (L*, a*, b*)": FROZEN_20[:15] + ["dark_region_mean_L"],
    "Segmentation": ["foreground_fraction"],
    "Cylindrical CIELAB (LCh)": ["h_std"],
    "CIEDE2000 perceptual colour similarity": FROZEN_20[16:20] + ["dE_margin_mean"],
}
FEATURE_HELP = {
    "Global Lab statistics (L*, a*, b*)": "Mean / median / spread / percentiles of the three CIELAB "
        "channels over the segmented banana, plus the mean L* of its dark pixels — the average "
        "colour and lightness, and how much they vary.",
    "Segmentation": "Fraction of the 416x416 frame occupied by the detected banana.",
    "Cylindrical CIELAB (LCh)": "h_ab = atan2(b*, a*) is the hue angle — the polar (LCh) form of the "
        "same a*/b* plane. Its standard deviation measures how mixed the banana's colour is "
        "(e.g. green regions + yellow regions give a wide hue spread).",
    "CIEDE2000 perceptual colour similarity": "ΔE00 from the banana's median colour to four fixed "
        "class reference colours (V1), plus the mean per-pixel gap between the nearest and "
        "second-nearest reference colour — small gap = the pixel's colour is perceptually ambiguous "
        "between two ripeness stages.",
}

# ------------------------------------------------------------------ OOD / reliability indicator
OOD_STATUS_TEXT = {
    "within":     ("Prediction reliability: within the validated image range",
                   "The image's colour features sit inside the distribution the model was validated on."),
    "borderline": ("Note: this image is near the edge of the validated range",
                   "Some colour features are unusual compared with the training data — treat the result with mild caution."),
    "outside":    ("Warning: this image appears outside the validated feature distribution",
                   "The image differs substantially from the training data (different banana variety, "
                   "lighting, or photo style). The prediction may be less reliable and is not overridden."),
}

# ------------------------------------------------------------------ UI copy
APP_TITLE = "Banana Ripeness Detection System"
APP_SUBTITLE = "Upload a photo to instantly check whether your banana is unripe, ripe, overripe or rotten"
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
MAX_UPLOAD_MB = 12
