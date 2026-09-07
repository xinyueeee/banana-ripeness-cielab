"""
media.py  --  batch (multi-image / folder) and video ingestion helpers
(Extra Efforts: "bulk image ingestion" and "video streams as input ... analysis
across individual frames").

Each item is run through the SAME single-image pipeline: decode -> 416x416 ->
frozen CIELAB segmentation -> 23 features -> StandardScaler+SVC -> class, plus the
object measurement and the reliability indicator. Nothing about the model changes.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config as cfg
from feature_extraction import extract_features
from model import predict, scaled_vector
from detection import measure


@dataclass
class ItemResult:
    name: str
    ok: bool
    prediction: str | None
    reliability: str | None
    foreground_pct: float | None
    length_px: float | None
    proc_ms: float
    note: str = ""
    frame_jpg: bytes | None = None      # for video frames: a small JPEG of the frame


def _frame_thumb_jpg(bgr: np.ndarray, width: int = 160) -> bytes:
    h, w = bgr.shape[:2]
    if w > width:
        bgr = cv2.resize(bgr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return buf.tobytes() if ok else b""


def _to_canvas(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != cfg.IMAGE_SIZE:
        rgb = cv2.resize(rgb, (cfg.IMAGE_SIZE[1], cfg.IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)
    return rgb


def analyse_rgb(rgb: np.ndarray, model, ood) -> ItemResult | tuple:
    """Internal: returns (ItemResult, fb, overlay_rgb) or ItemResult on failure."""
    t0 = time.perf_counter()
    fb = extract_features(rgb, model.prototypes)
    proc_ms = (time.perf_counter() - t0) * 1000.0
    if fb is None:
        return ItemResult("", False, None, None, None, None, proc_ms,
                          "segmentation produced an empty mask")
    pred = predict(model, fb.feature_vector)
    assessment = ood.assess(scaled_vector(model, fb.feature_vector))
    props = measure(fb.mask)
    res = ItemResult("", True, pred, assessment.status,
                     round(fb.foreground_fraction * 100, 1),
                     round(props.length_px, 0), round(proc_ms, 1))
    return res, fb, props


def process_images(files, model, ood, progress=None) -> list[ItemResult]:
    """files: iterable of (name, bytes). Returns one ItemResult per file."""
    results: list[ItemResult] = []
    files = list(files)
    for i, (name, data) in enumerate(files, 1):
        try:
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                results.append(ItemResult(name, False, None, None, None, None, 0.0,
                                          "could not decode image"))
            else:
                out = analyse_rgb(_to_canvas(bgr), model, ood)
                res = out if isinstance(out, ItemResult) else out[0]
                res.name = name
                results.append(res)
        except Exception as exc:  # noqa: BLE001
            results.append(ItemResult(name, False, None, None, None, None, 0.0, str(exc)))
        if progress:
            progress(i / len(files))
    return results


@dataclass
class VideoSummary:
    frames_analysed: int
    fps: float
    duration_s: float
    per_frame: list[ItemResult]
    majority_class: str | None
    class_counts: dict


def process_video(video_bytes: bytes, model, ood, *, sample_every_s: float = 1.0,
                  max_frames: int = 60, progress=None) -> VideoSummary:
    """Decode a video, sample one frame every `sample_every_s` seconds (capped at
    `max_frames`), and run the full pipeline on each sampled frame."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("The video could not be opened / decoded.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total / fps if fps else 0.0
        step = max(int(round(sample_every_s * fps)), 1)
        wanted = list(range(0, max(total, 1), step))[:max_frames]

        per_frame: list[ItemResult] = []
        for k, frame_idx in enumerate(wanted, 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            out = analyse_rgb(_to_canvas(frame), model, ood)
            res = out if isinstance(out, ItemResult) else out[0]
            res.name = f"t = {frame_idx / fps:5.1f} s"
            res.frame_jpg = _frame_thumb_jpg(frame)
            per_frame.append(res)
            if progress:
                progress(k / max(len(wanted), 1))
        cap.release()
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass

    good = [r.prediction for r in per_frame if r.ok and r.prediction]
    counts: dict = {}
    for c in good:
        counts[c] = counts.get(c, 0) + 1
    majority = max(counts, key=counts.get) if counts else None
    return VideoSummary(frames_analysed=len(per_frame), fps=float(fps),
                        duration_s=float(duration), per_frame=per_frame,
                        majority_class=majority, class_counts=counts)
