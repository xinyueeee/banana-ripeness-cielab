"""
consistency_check.py  --  self-contained integrity check for this prototype folder.

Verifies (needs only the files shipped in this folder):
  1. The trained StandardScaler + SVC reproduces the recorded official-test
     numbers on the bundled 23-feature test matrix.
  2. The feature pipeline is deterministic.
  3. If the Kaggle dataset happens to be reachable, that live feature extraction on a
     few test images matches the bundled feature matrix to numerical precision
     (this is the real "prototype == evaluated pipeline" proof; skipped otherwise).

Read-only. Not required to run the app.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import config as cfg
from model import load_model, predict
from feature_extraction import extract_features

meta = json.loads((APP_DIR / "models" / "metadata.json").read_text())
model = load_model()
tv = pd.read_csv(APP_DIR / "models" / "train_valid_features.csv.gz")
te = pd.read_csv(APP_DIR / "models" / "test_features.csv.gz")

assert list(tv.columns[1:]) == cfg.V3_FEATURES == list(te.columns[1:]), "feature-column mismatch"
assert len(tv) == 12916 and len(te) == 562, (len(tv), len(te))
assert model.n_features == len(cfg.V3_FEATURES) == 23

# --- 1. recorded metrics ----------------------------------------------------
Xte = te[cfg.V3_FEATURES].to_numpy(np.float64)
yte = te["ground_truth_class"].to_numpy(str)
pred = model.pipeline.predict(Xte).astype(str)
acc = float((pred == yte).mean())
print(f"bundled test matrix: accuracy {acc:.6f}  (recorded {meta['held_out_test_accuracy']:.6f})")
ok1 = abs(acc - meta["held_out_test_accuracy"]) < 1e-9
print(f"  metric reproduction: {'OK' if ok1 else 'MISMATCH'}")

# --- 2. determinism -------------------------------------------------------
import cv2
demo = np.full((416, 416, 3), 235, np.uint8)
cv2.ellipse(demo, (208, 208), (150, 55), 15, 0, 360, (60, 175, 215), -1)
demo = cv2.cvtColor(demo, cv2.COLOR_BGR2RGB)
f1 = extract_features(demo, model.prototypes).feature_vector
f2 = extract_features(demo, model.prototypes).feature_vector
ok2 = np.allclose(f1, f2, atol=1e-12) and f1.shape == (23,)
print(f"  deterministic feature extraction: {'OK' if ok2 else 'FAIL'}")

# --- 3. optional: live extraction vs bundled matrix ---------------------
DATASET = next((c for c in [
    Path(r"C:\Users\tanxi\Desktop\Y2S3\IP\Banana Ripeness Classification Dataset"),
    APP_DIR.parent / "Banana Ripeness Classification Dataset",
    APP_DIR.parent.parent / "Banana Ripeness Classification Dataset",
] if (c / "test").is_dir()), None)
ok3 = True
if DATASET is None:
    print("  live-extraction check: SKIPPED (dataset not found; not required)")
else:
    import glob
    max_d = 0.0
    for cls in cfg.CLASS_ORDER:
        p = sorted(glob.glob(str(DATASET / "test" / cls / "*")))[0]
        bgr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != cfg.IMAGE_SIZE:
            rgb = cv2.resize(rgb, cfg.IMAGE_SIZE[::-1], interpolation=cv2.INTER_AREA)
        fv = extract_features(rgb, model.prototypes).feature_vector
        cand = te[te.ground_truth_class == cls][cfg.V3_FEATURES].to_numpy(np.float64)
        d = float(np.abs(cand - fv).sum(axis=1).min())
        max_d = max(max_d, d)
    ok3 = max_d < 1e-6
    print(f"  live extraction vs bundled matrix: max abs diff over 4 images = {max_d:.2e}  "
          f"{'OK' if ok3 else 'MISMATCH'}")

print("\nRESULT:", "PASS" if (ok1 and ok2 and ok3) else "FAIL")
sys.exit(0 if (ok1 and ok2 and ok3) else 1)
