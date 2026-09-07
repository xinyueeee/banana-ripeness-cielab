"""
build_model.py  --  one-off: fit and save the CIELAB classifier + OOD reference.

Reproducible and self-contained: fits from the bundled pre-computed feature
matrices in models/ (produced by the FROZEN CIELAB pipeline). Needs no dataset
images, no notebook, no internet.

Classifier: StandardScaler() -> default SVC() -- the exact shared configuration
used by every approach in the group comparison. No hyperparameter is tuned and
no classification-stage change is applied.

Outputs (into models/):
    cielab_model.joblib   the fitted Pipeline([StandardScaler, SVC])
    ood_reference.joblib  scaled training matrix + validation p95/p99 distances
    metadata.json         provenance + verification numbers
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.validation import check_is_fitted

from config import (MODELS_DIR, MODEL_PATH, OOD_PATH, METADATA_PATH,
                    V3_FEATURES, V3_EXTRA, FROZEN_20, CLASS_ORDER)

TRAIN_VALID = MODELS_DIR / "train_valid_features.csv.gz"
TEST = MODELS_DIR / "test_features.csv.gz"
N_VALID = 1123     # bundle is train (11,793) then valid (1,123), in order


def make_model() -> Pipeline:
    """The group-agreed untuned classifier."""
    return Pipeline([("scaler", StandardScaler()), ("svm", SVC())])


def main() -> None:
    tv = pd.read_csv(TRAIN_VALID)
    assert list(tv.columns[1:]) == V3_FEATURES, "feature bundle column mismatch"
    assert len(tv) == 12916, len(tv)

    X = tv[V3_FEATURES].to_numpy(np.float64)
    y = tv["ground_truth_class"].to_numpy(str)
    X_train, X_valid = X[:-N_VALID], X[-N_VALID:]

    model = make_model().fit(X, y)
    check_is_fitted(model.named_steps["scaler"])
    check_is_fitted(model.named_steps["svm"])

    # OOD reference lives in the fitted scaler's space.
    scaler = model.named_steps["scaler"]
    train_scaled = scaler.transform(X_train)
    nn_train = NearestNeighbors(n_neighbors=1).fit(train_scaled)
    valid_dist = nn_train.kneighbors(scaler.transform(X_valid))[0].ravel()
    quantiles = {p: float(np.percentile(valid_dist, p)) for p in (50, 90, 95, 99, 99.5)}
    joblib.dump({"nn_train_scaled_X": train_scaled, "valid_nn_dist_quantiles": quantiles}, OOD_PATH)
    joblib.dump(model, MODEL_PATH)

    te = pd.read_csv(TEST)
    Xte, yte = te[V3_FEATURES].to_numpy(np.float64), te["ground_truth_class"].to_numpy(str)
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    mf1 = f1_score(yte, pred, average="macro", labels=CLASS_ORDER, zero_division=0)
    assert (joblib.load(MODEL_PATH).predict(Xte) == pred).all(), "reloaded model disagrees"

    METADATA_PATH.write_text(json.dumps({
        "version": "CIELAB (23-feature, plain SVC)",
        "classifier": "Pipeline([StandardScaler(), SVC()])  -- scikit-learn defaults, "
                      "RBF kernel, probability=False, no hyperparameter tuned",
        "feature_set": V3_FEATURES, "frozen_features": FROZEN_20, "added_features": V3_EXTRA,
        "n_features": len(V3_FEATURES),
        "feature_selection": "the 3 added descriptors were chosen by greedy forward 5-fold "
                             "CV macro-F1 on train + validation only",
        "fitted_on": "train + validation = 12,916 rows (frozen pipeline features)",
        "classes": sorted(np.unique(y).tolist()),
        "ood_thresholds_from_validation": {"p95": quantiles[95], "p99": quantiles[99]},
        "held_out_test_accuracy": float(acc), "held_out_test_macro_f1": float(mf1),
        "v1_frozen_test_accuracy": 0.9359430604982206,
        "note": "The 562-image test set was evaluated once. This rebuild reproduces that "
                "number from the bundled feature matrix.",
    }, indent=2))

    print(f"saved {MODEL_PATH.name}, {OOD_PATH.name}, {METADATA_PATH.name}")
    print(f"rebuild held-out test: accuracy={acc:.6f}  macroF1={mf1:.6f}  (V1 0.935943)")
    print(f"OOD thresholds (validation): p95={quantiles[95]:.3f}  p99={quantiles[99]:.3f}")
    print("reload check: OK")


if __name__ == "__main__":
    main()
