"""
v3_features.py  --  candidate V3 features: ENHANCED CIELAB / CIEDE2000 colour
representation ONLY. No texture, no blob analysis, no new colour space.

Two groups, both computed on the SAME frozen segmentation mask and the SAME
true-unit CIELAB channels used by the frozen pipeline:

  L  Cylindrical CIELAB (LCh)            10   C*_ab and h_ab are the polar form of
                                              the a*/b* plane -> still CIELAB.
  D  Per-pixel CIEDE2000 distributions   26   the frozen CIEDE2000 metric applied
                                              per pixel and summarised as a
                                              perceptual colour-composition, instead
                                              of V1's single mean-colour distance.

Plus one already-approved V1-style statistic:
  dark_region_mean_L                      1   conditional mean of L* (pure CIELAB).

Nothing here is LBP / GLCM / connected-component / grid / HSV.
"""
from __future__ import annotations

import numpy as np

from cielab_pipeline import ciede2000, CLASS_ORDER      # frozen CIEDE2000 implementation

# fixed thresholds (perceptual L*, device-independent) -----------------------
DARK_L = 40.0
PCTLS = (10, 25, 50, 75, 90)

# ---------------------------------------------------------------------------- names
LCH_NAMES = (
    ["C_mean", "C_median", "C_std", "C_p10", "C_p90"]
    + ["h_mean", "h_median", "h_std", "h_p10", "h_p90"]
)
_DE_PROTO_NAMES = [
    f"dE_{c}_p{p}" for c in CLASS_ORDER for p in PCTLS
]                                                        # 4 classes x 5 pctls = 20
_DE_COMP_NAMES = (
    [f"prop_nearest_{c}" for c in CLASS_ORDER]           # 4
    + ["dE_nearest_mean", "dE_margin_mean"]              # 2
)                                                        # 26 total
CIEDE_DIST_NAMES = _DE_PROTO_NAMES + _DE_COMP_NAMES

V3_CANDIDATE_NAMES: list[str] = LCH_NAMES + CIEDE_DIST_NAMES + ["dark_region_mean_L"]

V3_GROUPS: dict[str, list[str]] = {
    "L_lch": LCH_NAMES,
    "D_ciede_proto_pctl": _DE_PROTO_NAMES,
    "D_ciede_composition": _DE_COMP_NAMES,
    "S_safe_l_stat": ["dark_region_mean_L"],
}


def _pctl(values: np.ndarray, ps=PCTLS) -> list[float]:
    return [float(np.percentile(values, p)) for p in ps]


def extract_v3_features(
    L: np.ndarray, a: np.ndarray, b: np.ndarray, mask: np.ndarray,
    prototypes,
) -> dict[str, float]:
    """LCh + per-pixel CIEDE2000 distribution features for one segmented banana.

    `prototypes` is the frozen training-only DataFrame with columns
    class_name / prototype_L / prototype_a / prototype_b.
    """
    fg = mask > 0
    n_fg = int(fg.sum())
    if n_fg < 50:
        return {n: 0.0 for n in V3_CANDIDATE_NAMES}

    Lf, af, bf = L[fg], a[fg], b[fg]

    # ------------------------------------------------------------ group L : LCh
    C = np.hypot(af, bf)                                    # chroma  C*_ab
    h = np.degrees(np.arctan2(bf, af)) % 360.0              # hue angle h_ab (deg)
    # banana hues sit ~40-160 deg (no 0/360 wrap) so linear statistics are valid.
    lch = {
        "C_mean": float(C.mean()), "C_median": float(np.median(C)), "C_std": float(C.std()),
        "h_mean": float(h.mean()), "h_median": float(np.median(h)), "h_std": float(h.std()),
    }
    for name, arr in (("C", C), ("h", h)):
        p10, p25, p50, p75, p90 = _pctl(arr)
        lch[f"{name}_p10"] = p10
        lch[f"{name}_p90"] = p90

    # ---------------------------------------------- group D : per-pixel CIEDE2000
    px = np.stack([Lf, af, bf], axis=1).astype(np.float64)  # (n_fg, 3)
    proto_lab = {
        r.class_name: np.array([r.prototype_L, r.prototype_a, r.prototype_b], dtype=np.float64)
        for r in prototypes.itertuples()
    }
    dE = {}
    for c in CLASS_ORDER:
        ref = np.broadcast_to(proto_lab[c], px.shape)
        dE[c] = ciede2000(px, ref)                          # (n_fg,) frozen metric

    feats: dict[str, float] = dict(lch)
    for c in CLASS_ORDER:
        for p, v in zip(PCTLS, _pctl(dE[c])):
            feats[f"dE_{c}_p{p}"] = v

    stacked = np.stack([dE[c] for c in CLASS_ORDER], axis=1)  # (n_fg, 4)
    nearest_idx = np.argmin(stacked, axis=1)
    sorted_d = np.sort(stacked, axis=1)
    for i, c in enumerate(CLASS_ORDER):
        feats[f"prop_nearest_{c}"] = float((nearest_idx == i).mean())
    feats["dE_nearest_mean"] = float(sorted_d[:, 0].mean())
    feats["dE_margin_mean"] = float((sorted_d[:, 1] - sorted_d[:, 0]).mean())

    # ------------------------------------------------------------ safe L* stat
    dark = Lf < DARK_L
    feats["dark_region_mean_L"] = float(Lf[dark].mean()) if dark.any() else DARK_L

    return feats


def lch_maps_for_display(a: np.ndarray, b: np.ndarray):
    """(C*_ab map, h_ab map) for optional UI panels — same definitions as above."""
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360.0
    return C, h
