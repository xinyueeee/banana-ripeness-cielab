"""
model.py  --  load the trained CIELAB classifier and run predictions.

Classifier: Pipeline([StandardScaler(), SVC()]) -- the exact shared configuration
used by every approach in the group comparison (scikit-learn defaults, RBF
kernel, probability=False). No hyperparameter is tuned and no classification-stage
change is applied.

Fitted once by build_model.py on train + validation (12,916 rows) and saved to
models/cielab_model.joblib. Never retrained at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from config import MODEL_PATH, PROTOTYPES_PATH, V3_FEATURES


@dataclass
class LoadedModel:
    pipeline: Pipeline
    prototypes: pd.DataFrame
    n_features: int
    classes: list[str]


def load_model() -> LoadedModel:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. Run  python build_model.py  first."
        )
    pipe: Pipeline = joblib.load(MODEL_PATH)
    check_is_fitted(pipe.named_steps["scaler"])
    check_is_fitted(pipe.named_steps["svm"])
    svc = pipe.named_steps["svm"]
    if svc.n_features_in_ != len(V3_FEATURES):
        raise ValueError(
            f"Model expects {svc.n_features_in_} features but the pipeline produces {len(V3_FEATURES)}."
        )
    return LoadedModel(pipeline=pipe, prototypes=pd.read_csv(PROTOTYPES_PATH),
                       n_features=int(svc.n_features_in_),
                       classes=sorted(svc.classes_.tolist()))


def _as_row(feature_vector: np.ndarray) -> np.ndarray:
    x = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
    return x


def predict(model: LoadedModel, feature_vector: np.ndarray) -> str:
    x = _as_row(feature_vector)
    if x.shape[1] != model.n_features:
        raise ValueError(f"Expected {model.n_features} features, got {x.shape[1]}.")
    return str(model.pipeline.predict(x)[0])


def predict_with_trace(model: LoadedModel, feature_vector: np.ndarray) -> dict:
    """Final class plus the SVC decision-function margins per class, so the UI
    can show HOW the class was reached (these are margins, not probabilities)."""
    x = _as_row(feature_vector)
    if x.shape[1] != model.n_features:
        raise ValueError(f"Expected {model.n_features} features, got {x.shape[1]}.")
    svc = model.pipeline.named_steps["svm"]
    df = np.atleast_2d(model.pipeline.decision_function(x))[0]
    pred = str(model.pipeline.predict(x)[0])
    return {
        "final_pred": pred,
        "decision": {str(c): float(v) for c, v in zip(svc.classes_, df)},
    }


def scaled_vector(model: LoadedModel, feature_vector: np.ndarray) -> np.ndarray:
    """StandardScaler-transformed vector (the space the OOD detector's reference
    distances were built in)."""
    x = _as_row(feature_vector)
    return model.pipeline.named_steps["scaler"].transform(x)[0]
