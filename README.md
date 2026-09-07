# CIELAB Banana Ripeness — Prototype

Banana ripeness detection system and demo UI for the **CIELAB colour-space approach**.

| | |
|---|---|
| **Approach** | CIELAB colour segmentation + CIEDE2000 perceptual colour features |
| **Features** | **23** = 15 CIELAB channel statistics + foreground fraction + 4 CIEDE2000 prototype distances + 3 CIELAB / LCh / CIEDE2000 descriptors (`dark_region_mean_L`, `h_std`, `dE_margin_mean`) |
| **Model file** | `models/cielab_model.joblib` — `Pipeline([StandardScaler(), SVC()])` |
| **Classifier** | `StandardScaler` → scikit-learn `SVC()` (RBF, defaults, `probability=False`) — the exact shared configuration used by all five approaches in the group comparison, with **no hyperparameter tuned** and no classification-stage change |
| **Classes** | `unripe`, `ripe`, `overripe`, `rotten` |
| **Fitted on** | Kaggle **train + validation = 12,916** images (test set never used for fitting) |
| **Official test result** | accuracy **0.9484**, macro F1 **0.9490** (single evaluation of the frozen config) |

---

## What this prototype is

A demonstration of the CIELAB image-processing method that satisfies the assignment's
**Shared Core Functional Requirements**:

* **Preprocessing** — 3×3 median filter + CLAHE on `L*` (shown stage-by-stage).
* **Image calibration** — (1) *spatial scaling*: every image resized to a common 416×416
  canvas so pixel counts and area fractions are comparable; (2) *geometric rectification*:
  the banana's principal axis (min-area rectangle) is rotated to horizontal before
  measurement, so the bounding box, aspect ratio and extent are pose-independent.
  Rectification is measurement-only and does not change the 23 classifier features.
* **Object detection** — the banana contour and bounding box, read from the segmentation
  mask (Otsu on `b*` → morphology → largest connected component → bounded hole-fill).
* **Data-analysis dashboard** — predicted class, reliability indicator, object properties,
  the 23 feature values, the SVC decision margins, and every processing stage.

Extra-effort features: **PDF report export**, **bulk / multi-image analysis**, and
**video ingestion** (samples frames at a fixed interval and classifies each one).

**The prototype does not manually override predictions.** The class comes only from the
trained `SVC`. The out-of-distribution / reliability indicator is a **warning only** — it
never changes the predicted class (verified in `tests/`).

---

## How to run

```bash
cd CIELAB_banana_prototype
pip install -r requirements.txt

python build_model.py            # (re)builds models/ from the bundled feature matrices; ~10 s
python -m streamlit run app.py   # opens the UI at http://localhost:8501
```

Optional checks (no dataset needed):

```bash
python tests/test_prototype.py   # unit tests
python consistency_check.py      # proves the prototype pipeline == the evaluated pipeline
```

`build_model.py` only reads files inside `models/`. It fits the classifier from
`models/train_valid_features.csv.gz` (12,916 rows × 23 features, produced by the frozen
pipeline) and prints a verification accuracy against `models/test_features.csv.gz`
(562 rows). Neither CSV contains images; the test CSV is used only for that
post-build print, never for fitting.

---

## Required dependencies

```
streamlit  opencv-python(-headless)  numpy  pandas  scikit-learn  joblib  pillow  reportlab  xlsxwriter
```
(pinned in `requirements.txt`). Python 3.10+.

---

## File structure

```
CIELAB_banana_prototype/
├── app.py                 Streamlit UI (single-image / batch / video modes)
├── cielab_pipeline.py     frozen preprocessing + segmentation + CIEDE2000 + prototypes
├── v3_features.py         LCh + per-pixel CIEDE2000 descriptors
├── feature_extraction.py  builds the 23-feature vector for one image
├── model.py               load the model, predict, predict_with_trace
├── ood_detection.py       nearest-neighbour novelty indicator (warning only)
├── detection.py           bounding box + contour overlay + geometric rectification + object measurement
├── media.py               batch (multi-image) and video-frame ingestion
├── report_pdf.py          one-analysis PDF export
├── excel_export.py        multi-image results -> .xlsx with an embedded thumbnail per row
├── visualisation.py       UI panels / tables
├── config.py              every constant, threshold, feature name, label, UI string
├── build_model.py         one-off: fit + save models/ from the bundled feature matrices
├── consistency_check.py   prototype-vs-evaluated-pipeline integrity check
├── requirements.txt
├── AUDIT_REPORT.md
├── README.md              this file
├── tests/test_prototype.py
├── assets/                sample images + comparison figures
└── models/
    ├── cielab_model.joblib        the trained Pipeline([StandardScaler, SVC])
    ├── ood_reference.joblib       scaled training matrix + validation p95/p99 thresholds
    ├── training_prototypes.csv    4 CIEDE2000 class reference colours (TRAIN subset only)
    ├── metadata.json              provenance + recorded metrics
    ├── train_valid_features.csv.gz  12,916 × (label + 23 features)   [build input]
    └── test_features.csv.gz         562 × (label + 23 features)      [verification only]
```

---

## Expected input / output

**Input:** one JPG / JPEG / PNG / BMP / WEBP photo of a banana (single-image mode — file
upload **or the device camera**), several images at once (multiple-images mode), or a
short MP4/AVI/MOV clip (video mode). Any size; resized to 416 × 416. Best results with a
single banana that fills a reasonable part of the frame.

**Output:**

* **Predicted ripeness** — one of `UNRIPE / RIPE / OVERRIPE / ROTTEN`.
* **Reliability indicator** — *within the validated range* / *near the edge* / *outside the
  validated distribution*. Thresholds are the 95th / 99th percentile of the
  nearest-training-neighbour distance on the **validation set only**. Advisory; never
  overrides the class.
* **Detected object** — contour + bounding box overlay, and an object-properties table
  (bounding box, area, approximate length × width, aspect ratio, extent, perimeter,
  centroid — all in pixels).
* **Image-calibration summary**, the **23 feature values** grouped by type, the
  **SVC decision margins** per class (margins, not probabilities — the model has no
  probability output), and every processing stage.
* **PDF report** — a detailed one for a single analysis, and a **combined batch PDF**
  (class counts + one thumbnail row per image) for a multiple-image run.
* **CSV** and **Excel (.xlsx with an embedded thumbnail per row)** of the multiple-image
  *and* per-frame video results.
* **Video preview** player before analysis; a thumbnail of every sampled frame in the
  per-frame results table.
* **Suspicious-mask warning** if the segmented foreground fraction falls outside 0.02–0.90.

---

## Known limitations

### Overripe / Rotten dataset label definition (important)

In the Kaggle dataset used for this project:

* **Overripe** images are predominantly a **uniformly dark-brown / black peel**.
* **Rotten** images frequently **retain large yellow regions** together with structural
  decay (splitting, mould, collapsed tissue).

Consequently, a **real-world yellow banana with many brown spots** — which many people
would call *overripe* — falls inside **this dataset's *rotten* colour distribution** and
is predicted **ROTTEN**. This is a **characteristic of the dataset's labelling**, not a
bug and not a hidden rule: the classifier is faithfully reproducing what it was trained
on. The out-of-distribution warning is designed to flag exactly these images.

### Other limitations

* Trained on one dataset; one banana variety dominates the images.
* Segmentation assumes the banana is strongly yellow (positive `b*`) against a
  near-neutral background; unusual backgrounds can produce a poor mask (flagged as
  "suspicious").
* The prototype assumes the input is a single banana; other fruits or objects still
  receive a ripeness label, flagged by the reliability indicator as outside the
  validated range.
* External / Internet images frequently trigger the out-of-distribution warning — that
  is the indicator working as intended.
* The reliability indicator is a **distance heuristic**, not a calibrated probability.
* Object measurements are in **pixels** (416×416 canvas); the dataset provides no
  physical scale reference for millimetre calibration.

---

## Deploying (Streamlit Community Cloud)

`requirements.txt` and `.streamlit/config.toml` are set up for a one-click deploy: push
this folder to GitHub, then at share.streamlit.io point the app at `app.py`. No
`packages.txt` is needed -- `opencv-python-headless` bundles its own codecs.

`.gitignore` deliberately excludes `assets/mascot/` (a decorative third-party image). It
stays in the local folder and the submission ZIP, but is **not** pushed to the public
deployment repo; `hero_html` falls back to a plain banner when it is absent.

---

## Important statement

**This prototype does not manually override model predictions, does not contain
hard-coded or class-specific rules, and does not use any image-processing technique
outside the CIELAB / LCh / CIEDE2000 approach.** The bounding-box / contour overlay and
object measurements are read-outs of the segmentation mask that the CIELAB pipeline
already produces; none of them is fed to the classifier.
