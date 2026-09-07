"""
app.py  --  Banana Ripeness Detection System  .  CIELAB prototype (Streamlit UI).

Thin orchestration only. Work is delegated to:
    cielab_pipeline.py     frozen preprocessing + segmentation (verbatim from the assignment)
    feature_extraction.py  frozen 20 + 3 CIELAB / LCh / CIEDE2000 descriptors  (23 features)
    model.py               load StandardScaler+SVC + predict (cached)
    ood_detection.py       reliability / out-of-distribution indicator
    detection.py           bounding box + contour overlay + object measurement
    media.py               batch (multi-image) and video-frame ingestion
    report_pdf.py          one-analysis PDF export
    visualisation.py       display panels + tables
    config.py              every constant / threshold / label

Run:  python -m streamlit run app.py
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from feature_extraction import extract_features
from model import load_model, predict_with_trace, scaled_vector
from ood_detection import OODDetector
from detection import measure, detection_overlay, rectified_overlay, calibration_rows
from visualisation import (pipeline_stage_panels, result_badge_html,
                           feature_group_tables, processing_summary)
import media
import report_pdf
import excel_export
import theme

st.set_page_config(page_title="Banana Ripeness Detection System",
                   page_icon="🍌", layout="wide")
st.markdown(theme.inject_css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def get_resources():
    return load_model(), OODDetector()


def decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise ValueError("The file could not be read as an image.")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("The image does not have three colour channels.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != cfg.IMAGE_SIZE:
        rgb = cv2.resize(rgb, (cfg.IMAGE_SIZE[1], cfg.IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)
    return rgb


def analyse(rgb: np.ndarray):
    model, ood = get_resources()
    t0 = time.perf_counter()
    fb = extract_features(rgb, model.prototypes)
    proc_ms = (time.perf_counter() - t0) * 1000.0
    if fb is None:
        return None, None, None, (proc_ms, 0.0), None, None
    t1 = time.perf_counter()
    trace = predict_with_trace(model, fb.feature_vector)
    pred = trace["final_pred"]
    clf_ms = (time.perf_counter() - t1) * 1000.0
    assessment = ood.assess(scaled_vector(model, fb.feature_vector))
    props = measure(fb.mask)
    return fb, pred, assessment, (proc_ms, clf_ms), trace, props


# ----------------------------------------------------------------- header + sidebar
st.markdown(theme.hero_html(cfg.APP_TITLE, cfg.APP_SUBTITLE), unsafe_allow_html=True)
st.markdown(theme.class_legend_html(), unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🍌 Quick guide")
    st.markdown(
        "1. Pick a mode below.\n"
        "2. Upload one banana photo, several photos, or a short video.\n"
        "3. Read the predicted class, the reliability flag, the detected object, "
        "and every processing stage.\n"
    )
    st.markdown("### Ripeness classes")
    for c in ["unripe", "ripe", "overripe", "rotten"]:
        st.markdown(
            f"<span class='chip' style='background:{cfg.CLASS_COLOUR[c]}'>"
            f"{theme.CLASS_EMOJI[c]} {c.upper()}</span>", unsafe_allow_html=True)
    st.markdown("")
    st.caption("Colour-only prototype: no texture, shape or deep learning -- only "
               "CIELAB / LCh / CIEDE2000 colour features and a StandardScaler + SVC classifier.")

try:
    model, _ = get_resources()
    model_ready = True
except Exception as exc:  # noqa: BLE001
    model_ready = False
    st.error("The trained model could not be loaded.")
    st.info(str(exc))

st.divider()
mode = st.radio("Mode", ["Single image", "Multiple images", "Video"],
                horizontal=True, label_visibility="collapsed")


def _thumb_uri(data: bytes, px: int = 72) -> str:
    """Small base64 JPEG data URI for a results-table preview column."""
    import base64
    from io import BytesIO
    from PIL import Image
    try:
        im = Image.open(BytesIO(data)).convert("RGB")
        im.thumbnail((px, px))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return ""


# ================================================================= SINGLE IMAGE
def render_single():
    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("1 . Provide a banana image")
            src = st.radio("Image source", ["Upload file", "Use camera"],
                           horizontal=True, key="single_src", label_visibility="collapsed")
            if src == "Use camera":
                upload = st.camera_input("Take a photo of a banana", key="single_cam")
            else:
                upload = st.file_uploader("Upload Banana Image", type=cfg.SUPPORTED_TYPES,
                                          key="single_up")
            c1, c2 = st.columns(2)
            go = c1.button("Analyse Ripeness", type="primary", use_container_width=True,
                           disabled=(upload is None or not model_ready))
            if c2.button("Reset / New Image", use_container_width=True):
                st.session_state.pop("result", None)
                st.rerun()
            if upload is not None:
                if upload.size > cfg.MAX_UPLOAD_MB * 1024 * 1024:
                    st.warning(f"File larger than {cfg.MAX_UPLOAD_MB} MB - please use a smaller image.")
                elif src != "Use camera":
                    st.image(upload, caption="Original image", width=320)
            elif src == "Use camera":
                st.info("Allow camera access and take a photo of a single banana.")
            else:
                st.info("Upload a JPG, JPEG, PNG, BMP or WEBP image of a banana to begin.")

    if go and upload is not None and model_ready:
        try:
            rgb = decode_upload(upload.getvalue())
        except ValueError as exc:
            st.session_state["result"] = {"error": f"Invalid or unreadable image: {exc}"}
        else:
            try:
                fb, pred, assessment, (proc_ms, clf_ms), trace, props = analyse(rgb)
                if fb is None:
                    st.session_state["result"] = {
                        "error": "The banana could not be segmented (empty mask). "
                                 "Try a clearer, closer photo of a single banana."}
                else:
                    st.session_state["result"] = {
                        "name": ("camera photo" if src == "Use camera"
                                 else getattr(upload, "name", "image")),
                        "fb": fb, "pred": pred, "ood": assessment,
                        "proc_ms": proc_ms, "clf_ms": clf_ms, "trace": trace, "props": props}
            except Exception as exc:  # noqa: BLE001
                st.session_state["result"] = {"error": "Image processing failed.", "detail": str(exc)}

    with right:
        with st.container(border=True):
            st.subheader("2 . Result")
            res = st.session_state.get("result")
            if res is None:
                st.info("Upload an image and press **Analyse Ripeness**.")
            elif "error" in res:
                st.error(res["error"])
                if res.get("detail"):
                    st.caption(res["detail"])
            else:
                fb, pred, ood, props = res["fb"], res["pred"], res["ood"], res["props"]
                st.markdown("**Predicted Ripeness**")
                st.markdown(result_badge_html(pred), unsafe_allow_html=True)
                st.caption("Class produced by the trained StandardScaler + SVC on the 23 CIELAB / "
                           "LCh / CIEDE2000 features of this image. The SVC has no probability "
                           "model, so no confidence percentage is shown.")
                (st.success if ood.status == "within" else st.warning)(ood.headline)
                st.caption(f"{ood.detail}  (novelty distance {ood.distance:.2f}; "
                           f"validated <= {ood.p95:.2f}, edge <= {ood.p99:.2f}).")
                if fb.suspicious_mask:
                    st.warning("The banana region could not be confidently segmented "
                               f"(foreground fraction {fb.foreground_fraction:.2f}). "
                               "Check the mask in the Image Processing tab.")
                st.image(detection_overlay(fb.rgb_resized, fb.mask, pred),
                         caption="Detected banana: contour (green) + bounding box (blue)",
                         use_container_width=True)

    res = st.session_state.get("result")
    if not (res and "error" not in res):
        return
    fb = res["fb"]
    tabs = st.tabs(["Result", "Image Processing", "Feature Analysis",
                    "Detection & Calibration", "How It Works", "Export"])

    with tabs[0]:
        st.table(processing_summary(fb, res["proc_ms"], res["clf_ms"], model.n_features))
        tr = res.get("trace")
        if tr:
            st.markdown("#### SVC decision margins")
            st.caption("Signed distance of this image from each class boundary (one-vs-one "
                       "aggregated). These are SVC margins, not probabilities; the predicted "
                       "class is the arg-max.")
            st.dataframe(pd.DataFrame(
                [{"class": k, "margin": round(v, 3)} for k, v in tr["decision"].items()]
            ).sort_values("margin", ascending=False),
                use_container_width=True, hide_index=True)
        st.markdown("**Visual pipeline:** Original -> CIELAB -> Banana Mask -> "
                    "Extracted Banana -> Detection -> Prediction")
        panels = pipeline_stage_panels(fb)
        cols = st.columns(len(panels))
        for col, (cap, img, _cmap) in zip(cols, panels):
            col.image(img, caption=cap, use_container_width=True, clamp=True)

    with tabs[1]:
        st.markdown("#### Processing stages")
        panels = pipeline_stage_panels(fb)
        for i in range(0, len(panels), 3):
            cols = st.columns(3)
            for col, (cap, img, _cmap) in zip(cols, panels[i:i + 3]):
                col.image(img, caption=cap, use_container_width=True, clamp=True)
        st.caption("C*ab and h_ab are the cylindrical (LCh) form of the same a*/b* plane -- "
                   "they are CIELAB, not a separate colour space.")

    with tabs[2]:
        st.markdown("The model uses **23 features** in four groups: 15 global CIELAB "
                    "statistics, the foreground fraction, four CIEDE2000 prototype distances, "
                    "and three additional CIELAB / LCh / CIEDE2000 descriptors "
                    "(`dark_region_mean_L`, `h_std`, `dE_margin_mean`).")
        for name, help_text, table in feature_group_tables(fb):
            st.markdown(f"**{name}**")
            st.caption(help_text)
            st.dataframe(table, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.markdown("#### Object detection")
        st.caption("The bounding box and contour are read from the banana mask that the "
                   "CIELAB segmentation already produced (Otsu on b* -> morphology -> "
                   "largest connected component -> bounded hole-fill). No extra detector "
                   "is used and none of these measurements is fed to the classifier.")
        rect_img, rot = rectified_overlay(fb.rgb_resized, fb.mask, res["pred"])
        c1, c2, c3 = st.columns(3)
        c1.image(detection_overlay(fb.rgb_resized, fb.mask, res["pred"]),
                 caption="As detected: contour + bounding box", use_container_width=True)
        c2.image(fb.mask, caption="Binary banana mask", use_container_width=True, clamp=True)
        c3.image(rect_img, caption=f"Rectified ({rot:+.1f} deg): principal axis levelled",
                 use_container_width=True)
        st.markdown("**Object properties** (measured on the rectified banana)")
        st.table(pd.DataFrame(res["props"].as_table_rows(), columns=["Property", "Value"]
                              ).set_index("Property"))
        st.markdown("#### Image calibration")
        st.table(pd.DataFrame(calibration_rows(res["props"].rotation_deg),
                              columns=["Parameter", "Value"]).set_index("Parameter"))
        st.caption(cfg.CALIBRATION_NOTE)

    with tabs[4]:
        st.markdown(
            """
1. **Resize** to 416 x 416 (INTER_AREA) and **3 x 3 median filter** -- removes speckle noise,
   preserves edges, and gives every image the same spatial scale. Together with the
   **geometric rectification** in step 5b this is the *image calibration*.
2. **RGB -> CIELAB** (true units) -- separates lightness (L\\*) from colour (a\\*, b\\*).
3. **CLAHE on L\\* only** -- local contrast for segmentation without shifting colour.
4. **Otsu threshold on b\\*** -- isolates the banana (strongly positive on the blue<->yellow axis).
5. **Morphology + largest connected component + bounded hole filling** -- clean mask; its
   contour and bounding box are the *object detection* output.
5b. **Geometric rectification** -- the banana's principal axis (from its min-area rectangle)
   is rotated to horizontal, so the bounding box, aspect ratio and extent are the same
   regardless of how the banana was oriented. Measurement only -- the 23 features below are
   still computed on the un-rectified frozen pipeline, so accuracy is unchanged.
6. **Features (23):** 15 global L\\*/a\\*/b\\* statistics + foreground fraction + four CIEDE2000
   prototype distances *(the frozen 20)*, plus **h_std** (LCh hue-angle spread),
   **dE_margin_mean** (per-pixel nearest / second-nearest CIEDE2000 gap) and
   **dark_region_mean_L** (mean L\\* of the dark pixels). The three extra descriptors were
   chosen by greedy forward 5-fold cross-validation on train + validation only.
7. **StandardScaler -> SVC()** -- the exact shared classifier used by all five approaches in
   the group comparison; no hyperparameter is tuned and no classification-stage change is applied.
8. A **nearest-neighbour reliability check** (thresholds from the validation set) flags images
   outside the validated range. It never changes the class.

On the untouched 562-image test set this configuration reaches **accuracy 0.9484** and
**macro F1 0.9490**.
            """
        )

    with tabs[5]:
        st.markdown("#### Export this analysis")
        st.caption("Automated PDF report: prediction, detection overlay, object properties, "
                   "image-calibration summary, processing summary and all 23 feature values.")
        feat_rows = [(n, f"{float(fb.feature_dict.get(n, float('nan'))):.4f}")
                     for n in cfg.V3_FEATURES]
        try:
            pdf_bytes = report_pdf.build_report(
                source_name=res["name"],
                rgb_resized=fb.rgb_resized,
                overlay_rgb=detection_overlay(fb.rgb_resized, fb.mask, res["pred"]),
                mask=fb.mask, prediction=res["pred"],
                ood_headline=res["ood"].headline, ood_detail=res["ood"].detail,
                object_rows=res["props"].as_table_rows(),
                calibration_rows=calibration_rows(res["props"].rotation_deg),
                feature_rows=feat_rows,
                processing_rows=[(i, r) for i, r in
                                 processing_summary(fb, res["proc_ms"], res["clf_ms"],
                                                    model.n_features).itertuples()],
            )
            st.download_button("Download PDF report", data=pdf_bytes,
                               file_name=f"banana_ripeness_{res['pred']}.pdf",
                               mime="application/pdf", type="primary")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not build the PDF: {exc}")


# ================================================================= MULTIPLE IMAGES
def render_batch():
    with st.container(border=True):
        st.subheader("Multiple-image analysis")
        st.caption("Select several images at once (hold Ctrl / Shift in the file dialog). "
                   "Every image goes through the identical single-image pipeline.")
        ups = st.file_uploader("Upload images", type=cfg.SUPPORTED_TYPES,
                               accept_multiple_files=True, key="batch_up")
        if ups:
            with st.expander(f"Preview {len(ups)} selected image(s)", expanded=False):
                per_row = 6
                for start in range(0, len(ups), per_row):
                    row = st.columns(per_row)
                    for slot, u in zip(row, ups[start:start + per_row]):
                        name = u.name if len(u.name) <= 22 else u.name[:19] + "…"
                        slot.image(u.getvalue(), width=110)
                        slot.caption(name)
        run = st.button("Analyse all", type="primary",
                        disabled=(not ups or not model_ready))
    if not (run and ups and model_ready):
        return
    _, ood = get_resources()
    bar = st.progress(0.0, text="Analysing...")
    payload = [(u.name, u.getvalue()) for u in ups]
    results = media.process_images(payload, model, ood, progress=lambda f: bar.progress(f))
    bar.empty()
    thumbs = {name: _thumb_uri(data) for name, data in payload}

    df = pd.DataFrame([{
        "preview": thumbs.get(r.name, ""),
        "image": r.name, "prediction": r.prediction or "-",
        "reliability": r.reliability or "-",
        "foreground %": r.foreground_pct, "length px": r.length_px,
        "time ms": r.proc_ms, "status": "ok" if r.ok else r.note,
    } for r in results])
    ok = df[df["status"] == "ok"]
    st.markdown(f"**{len(ok)} / {len(df)} images analysed**")
    if not ok.empty:
        counts = ok["prediction"].value_counts()
        cols = st.columns(len(cfg.CLASS_ORDER))
        for col, c in zip(cols, cfg.CLASS_ORDER):
            col.metric(f"{theme.CLASS_EMOJI[c]} {c}", int(counts.get(c, 0)))
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"preview": st.column_config.ImageColumn(
                     "preview", help="the uploaded image", width="small")})

    data_only = df.drop(columns=["preview"])
    records = data_only.to_dict("records")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Results (CSV)", data_only.to_csv(index=False).encode(),
                       file_name="banana_batch_results.csv", mime="text/csv",
                       use_container_width=True)
    try:
        xlsx = excel_export.build_xlsx(records, dict(payload), sheet_name="Banana results")
        d2.download_button("Excel (with thumbnails)", xlsx,
                           file_name="banana_batch_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        d2.caption(f"Excel unavailable: {exc}")
    try:
        pdf = report_pdf.build_batch_report(records=records, images=dict(payload))
        d3.download_button("Combined PDF report", pdf, file_name="banana_batch_report.pdf",
                           mime="application/pdf", type="primary", use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        d3.caption(f"PDF unavailable: {exc}")


# ================================================================= VIDEO
def render_video():
    with st.container(border=True):
        st.subheader("Video analysis")
        st.caption("Upload a short clip of a banana. The system samples frames at a fixed "
                   "interval and runs the full CIELAB pipeline on each sampled frame, then "
                   "reports the per-frame class and the overall majority vote.")
        vid = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "webm"],
                               key="video_up")
        if vid is not None:
            with st.expander("Preview video", expanded=True):
                st.video(vid)
        c1, c2 = st.columns(2)
        every = c1.slider("Sample one frame every (seconds)", 0.25, 5.0, 1.0, 0.25)
        cap_n = c2.slider("Max frames to analyse", 5, 120, 40, 5)
        run = st.button("Analyse video", type="primary",
                        disabled=(vid is None or not model_ready))
    if not (run and vid is not None and model_ready):
        return
    _, ood = get_resources()
    bar = st.progress(0.0, text="Sampling and analysing frames...")
    try:
        summary = media.process_video(vid.getvalue(), model, ood,
                                      sample_every_s=every, max_frames=cap_n,
                                      progress=lambda f: bar.progress(f))
    except Exception as exc:  # noqa: BLE001
        bar.empty()
        st.error(f"Video processing failed: {exc}")
        return
    bar.empty()

    a, b, c = st.columns(3)
    a.metric("Frames analysed", summary.frames_analysed)
    b.metric("Duration", f"{summary.duration_s:.1f} s")
    c.metric("Majority class",
             summary.majority_class.upper() if summary.majority_class else "-")

    if summary.class_counts:
        st.bar_chart(pd.Series(summary.class_counts).reindex(cfg.CLASS_ORDER).fillna(0))

    import base64
    frames = {r.name: (r.frame_jpg or b"") for r in summary.per_frame}
    df = pd.DataFrame([{
        "preview": ("data:image/jpeg;base64," + base64.b64encode(r.frame_jpg).decode()
                    if r.frame_jpg else ""),
        "frame": r.name, "prediction": r.prediction or "-",
        "reliability": r.reliability or "-", "foreground %": r.foreground_pct,
        "time ms": r.proc_ms, "status": "ok" if r.ok else r.note,
    } for r in summary.per_frame])
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"preview": st.column_config.ImageColumn(
                     "preview", help="the sampled frame", width="small")})

    data_only = df.drop(columns=["preview"])
    d1, d2 = st.columns(2)
    d1.download_button("Download per-frame results (CSV)", data_only.to_csv(index=False).encode(),
                       file_name="banana_video_results.csv", mime="text/csv",
                       use_container_width=True)
    try:
        xlsx = excel_export.build_xlsx(data_only.to_dict("records"), frames,
                                       sheet_name="Video frames", image_key="frame")
        d2.download_button("Download Excel (with frame thumbnails)", xlsx,
                           file_name="banana_video_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary", use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        d2.caption(f"Excel export unavailable: {exc}")


if mode == "Single image":
    render_single()
elif mode == "Multiple images":
    render_batch()
else:
    render_video()

st.divider()
with st.expander("Model status"):
    if model_ready:
        st.write({"features": model.n_features, "classes": model.classes,
                  "classifier": "Pipeline([StandardScaler(), SVC()])  (defaults, probability=False)"})
        try:
            import json
            st.json(json.loads(cfg.METADATA_PATH.read_text()))
        except Exception:  # noqa: BLE001
            pass
    else:
        st.write("model not loaded")

st.markdown(theme.footer_html(), unsafe_allow_html=True)
