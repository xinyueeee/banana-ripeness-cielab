"""
test_prototype.py  --  basic automated tests for the CIELAB V3 prototype.

Run:  python -m pytest tests/ -q      (from the CIELAB_Prototype_V3 folder)
or:   python tests/test_prototype.py  (plain runner, no pytest needed)

Covers: image loading, feature extraction + count, determinism, model loading,
prediction, segmentation, OOD detection, invalid-input handling.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import cv2

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

import config as cfg
from feature_extraction import extract_features, FeatureBundle
from model import load_model, predict, predict_with_trace, scaled_vector
from ood_detection import OODDetector

SAMPLE_DIR = APP_DIR / "assets" / "samples"


def _a_banana_rgb() -> np.ndarray:
    """A synthetic yellow banana on white — enough for the pipeline to segment."""
    img = np.full((416, 416, 3), 245, np.uint8)
    cv2.ellipse(img, (208, 208), (150, 55), 20, 0, 360, (60, 180, 220), -1)  # BGR-ish yellow
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _real_sample() -> np.ndarray | None:
    imgs = sorted(SAMPLE_DIR.glob("*")) if SAMPLE_DIR.is_dir() else []
    if not imgs:
        return None
    bgr = cv2.imdecode(np.fromfile(imgs[0], np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
def test_model_loads_and_is_fitted():
    m = load_model()
    assert m.n_features == len(cfg.V3_FEATURES) == 23
    assert sorted(m.classes) == sorted(cfg.CLASS_ORDER)


def test_feature_extraction_count_and_order():
    fb = extract_features(_a_banana_rgb(), load_model().prototypes)
    assert fb is not None
    assert fb.feature_vector.shape == (23,)
    assert np.isfinite(fb.feature_vector).all()
    # vector order must match config.V3_FEATURES exactly
    for i, name in enumerate(cfg.V3_FEATURES):
        assert abs(fb.feature_vector[i] - fb.feature_dict[name]) < 1e-9


def test_feature_extraction_is_deterministic():
    protos = load_model().prototypes
    rgb = _a_banana_rgb()
    v1 = extract_features(rgb, protos).feature_vector
    v2 = extract_features(rgb, protos).feature_vector
    assert np.allclose(v1, v2, atol=1e-12)


def test_segmentation_produces_a_mask():
    fb = extract_features(_a_banana_rgb(), load_model().prototypes)
    assert fb.mask.shape == (416, 416)
    assert 0.0 < fb.foreground_fraction < 1.0
    assert fb.mask.dtype == np.uint8 and set(np.unique(fb.mask)).issubset({0, 255})


def test_prediction_returns_valid_class():
    m = load_model()
    fb = extract_features(_a_banana_rgb(), m.prototypes)
    pred = predict(m, fb.feature_vector)
    assert pred in cfg.CLASS_ORDER


def test_ood_detector_runs_and_is_bounded():
    m = load_model()
    ood = OODDetector()
    fb = extract_features(_a_banana_rgb(), m.prototypes)
    a = ood.assess(scaled_vector(m, fb.feature_vector))
    assert a.status in {"within", "borderline", "outside"}
    assert a.distance >= 0.0 and a.p95 < a.p99


def test_trace_matches_prediction():
    """The trace's final_pred must equal predict(), and margins cover all 4 classes."""
    m = load_model()
    fb = extract_features(_a_banana_rgb(), m.prototypes)
    tr = predict_with_trace(m, fb.feature_vector)
    assert tr["final_pred"] in cfg.CLASS_ORDER
    assert tr["final_pred"] == predict(m, fb.feature_vector)
    assert set(tr["decision"]) == set(cfg.CLASS_ORDER)
    # arg-max of the margins is the predicted class
    assert max(tr["decision"], key=tr["decision"].get) == tr["final_pred"]


def test_object_measurement_and_overlay():
    from detection import measure, detection_overlay
    m = load_model()
    fb = extract_features(_a_banana_rgb(), m.prototypes)
    props = measure(fb.mask)
    assert props.detected and props.area_px > 0
    assert 0 < props.bbox_w <= 416 and 0 < props.bbox_h <= 416
    assert 0.0 < props.frame_fraction_pct <= 100.0
    ov = detection_overlay(fb.rgb_resized, fb.mask, "ripe")
    assert ov.shape == (416, 416, 3) and ov.dtype == np.uint8


def test_wrong_feature_count_is_rejected():
    m = load_model()
    try:
        predict(m, np.zeros(9))
    except ValueError:
        return
    raise AssertionError("predict() accepted a 10-D vector")


def test_empty_or_black_image_is_handled():
    # a pure-black frame -> segmentation should yield no banana -> None, not a crash
    black = np.zeros((416, 416, 3), np.uint8)
    fb = extract_features(black, load_model().prototypes)
    assert fb is None or fb.feature_vector.shape == (23,)


def test_real_sample_if_available():
    rgb = _real_sample()
    if rgb is None:
        return
    m = load_model()
    fb = extract_features(rgb, m.prototypes)
    assert fb is not None and predict(m, fb.feature_vector) in cfg.CLASS_ORDER


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
