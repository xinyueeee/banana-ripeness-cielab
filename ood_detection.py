"""
ood_detection.py  --  lightweight out-of-distribution / novelty indicator.

Why: the formal V1 diagnosis showed that some external images sit 4-10x farther
from the training feature envelope than any dataset test image, and the RBF SVC
still returns a confident-looking class for them. This module reports whether an
uploaded image's features are inside the range the model was validated on.

Method: distance to the nearest TRAINING sample in scaled 23-D feature space.
Thresholds are the 95th / 99th percentile of that distance measured on the
VALIDATION set only (stored in models/ood_reference.joblib by build_model.py).
The external test images were NOT used to choose the thresholds.

This is a diagnostic indicator, NOT a calibrated probability. It never changes
the predicted class.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

from config import OOD_PATH, OOD_STATUS_TEXT


@dataclass
class OODAssessment:
    status: str                  # "within" | "borderline" | "outside"
    distance: float              # nearest-training-neighbour distance (scaled space)
    p95: float
    p99: float
    headline: str
    detail: str


class OODDetector:
    def __init__(self) -> None:
        if not OOD_PATH.exists():
            raise FileNotFoundError(
                f"OOD reference not found: {OOD_PATH.name}. Run  python build_model.py  first."
            )
        ref = joblib.load(OOD_PATH)
        self._nn = NearestNeighbors(n_neighbors=1).fit(ref["nn_train_scaled_X"])
        q = ref["valid_nn_dist_quantiles"]
        self.p95 = float(q[95])
        self.p99 = float(q[99])

    def assess(self, scaled_feature_vector: np.ndarray) -> OODAssessment:
        d = float(self._nn.kneighbors(
            np.asarray(scaled_feature_vector, dtype=np.float64).reshape(1, -1)
        )[0].ravel()[0])
        status = "within" if d <= self.p95 else ("borderline" if d <= self.p99 else "outside")
        headline, detail = OOD_STATUS_TEXT[status]
        return OODAssessment(status=status, distance=d, p95=self.p95, p99=self.p99,
                             headline=headline, detail=detail)
