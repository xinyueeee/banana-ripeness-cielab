"""
report_pdf.py  --  automated export of results to a PDF report (Extra Effort:
"Automated export of results and findings into PDF format").

build_report(...)        -> bytes : one detailed single-image analysis
build_batch_report(...)  -> bytes : one combined report for a multi-image run

Pure functions. Uses reportlab + PIL only.
"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image as RLImage, PageBreak)

import config as cfg

_BANANA_YELLOW = colors.HexColor("#F2A900")
_INK = colors.HexColor("#4A3220")

_WRAP = ParagraphStyle("wrap", fontName="Helvetica", fontSize=8, leading=9.5,
                       textColor=_INK, wordWrap="CJK")     # CJK = break anywhere
_WRAP_B = ParagraphStyle("wrapb", parent=_WRAP, fontName="Helvetica-Bold")


def _p(text, bold: bool = False) -> Paragraph:
    """A wrapping cell: breaks long unbroken strings (filenames) instead of overflowing."""
    return Paragraph(str(text).replace("&", "&amp;").replace("<", "&lt;"),
                     _WRAP_B if bold else _WRAP)


def _png(rgb: np.ndarray) -> io.BytesIO:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb).astype("uint8")).save(buf, format="PNG")
    buf.seek(0)
    return buf


def _kv_table(rows, col_widths=(70 * mm, 95 * mm)):
    t = Table([[_p(k, bold=True), _p(v)] for k, v in rows], colWidths=list(col_widths))
    t.setStyle(TableStyle([
        ("TEXTCOLOR", (0, 0), (-1, -1), _INK),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#E5D3A8")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#FFFBF0")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build_report(*, source_name: str, rgb_resized: np.ndarray, overlay_rgb: np.ndarray,
                 mask: np.ndarray, prediction: str, ood_headline: str, ood_detail: str,
                 object_rows, calibration_rows, feature_rows,
                 processing_rows) -> bytes:
    """Return a PDF (bytes) summarising one banana-ripeness analysis."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=16 * mm, bottomMargin=16 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            title="Banana Ripeness Analysis Report")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=_INK, fontSize=18, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=_INK, fontSize=12,
                        spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], textColor=_INK, fontSize=9.5,
                          leading=13)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.HexColor("#8A6A46"))

    story = []
    story.append(Paragraph("Banana Ripeness Analysis Report", h1))
    story.append(Paragraph(
        f"CIELAB colour-space approach &nbsp;|&nbsp; StandardScaler + SVC "
        f"&nbsp;|&nbsp; generated {datetime.now():%Y-%m-%d %H:%M}", small))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Result", h2))
    story.append(_kv_table([
        ("Source", source_name),
        ("Predicted ripeness", prediction.upper()),
        ("Reliability", ood_headline),
    ]))
    story.append(Spacer(1, 2))
    story.append(Paragraph(ood_detail, small))

    # images side by side
    story.append(Paragraph("Detection", h2))
    img_row = Table([[
        RLImage(_png(rgb_resized), width=75 * mm, height=75 * mm),
        RLImage(_png(overlay_rgb), width=75 * mm, height=75 * mm),
    ]], colWidths=[82 * mm, 82 * mm])
    img_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(img_row)
    story.append(Paragraph("Left: prepared 416 x 416 input.&nbsp; Right: banana contour "
                           "(green) and bounding box (blue) from the CIELAB segmentation mask.",
                           small))

    story.append(Paragraph("Object properties", h2))
    story.append(_kv_table(list(object_rows)))

    story.append(Paragraph("Image calibration", h2))
    story.append(_kv_table(list(calibration_rows)))
    story.append(Paragraph(cfg.CALIBRATION_NOTE, small))

    story.append(PageBreak())
    story.append(Paragraph("Processing summary", h2))
    story.append(_kv_table(list(processing_rows)))

    story.append(Paragraph("Feature values (23)", h2))
    fr = [("Feature", "Value")] + [(str(a), str(b)) for a, b in feature_rows]
    ft = Table(fr, colWidths=[95 * mm, 70 * mm])
    ft.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("BACKGROUND", (0, 0), (-1, 0), _BANANA_YELLOW),
        ("TEXTCOLOR", (0, 0), (-1, -1), _INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFFBF0")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E5D3A8")),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(ft)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "This prototype demonstrates the CIELAB image-processing method. The class comes only "
        "from the trained SVC; the reliability indicator is advisory and never changes the class.",
        small))

    doc.build(story)
    return buf.getvalue()


def _thumb(data: bytes, px: int = 90) -> io.BytesIO | None:
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((px, px))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except Exception:  # noqa: BLE001
        return None


def build_batch_report(*, records: list[dict], images: dict[str, bytes],
                       title: str = "Banana Ripeness — Batch Report",
                       image_key: str = "image") -> bytes:
    """One combined PDF for a multi-image run: a summary line, a class-count table,
    and a table with one thumbnail + result row per image."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=16 * mm, bottomMargin=16 * mm,
                            leftMargin=16 * mm, rightMargin=16 * mm, title=title)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=_INK, fontSize=17, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=_INK, fontSize=12,
                        spaceBefore=10, spaceAfter=4)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8,
                           textColor=colors.HexColor("#8A6A46"), leading=11)

    ok = [r for r in records if str(r.get("status", "")).lower() == "ok"]
    counts: dict = {}
    for r in ok:
        counts[r.get("prediction", "-")] = counts.get(r.get("prediction", "-"), 0) + 1

    story = [Paragraph(title, h1)]
    story.append(Paragraph(
        f"CIELAB colour-space approach &nbsp;|&nbsp; StandardScaler + SVC &nbsp;|&nbsp; "
        f"generated {datetime.now():%Y-%m-%d %H:%M} &nbsp;|&nbsp; {len(ok)} / {len(records)} "
        f"images analysed", small))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Class counts", h2))
    story.append(_kv_table([(cls, str(counts.get(cls, 0))) for cls in cfg.CLASS_ORDER],
                           col_widths=(50 * mm, 30 * mm)))

    story.append(Paragraph("Per-image results", h2))
    cols = ["image", "prediction", "reliability", "foreground %", "length px"]
    head = [_p(h, bold=True) for h in ["preview"] + cols]
    data = [head]
    for r in records:
        buf_im = _thumb(images.get(r.get(image_key, ""), b""))
        cell0 = RLImage(buf_im, width=15 * mm, height=15 * mm) if buf_im else _p("n/a")
        data.append([cell0] + [_p(r.get(c, "-")) for c in cols])
    t = Table(data, colWidths=[19 * mm, 55 * mm, 20 * mm, 22 * mm, 22 * mm, 20 * mm],
              repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _BANANA_YELLOW),
        ("TEXTCOLOR", (0, 0), (-1, -1), _INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFFBF0")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E5D3A8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Every image was processed by the identical CIELAB pipeline (416x416 -> median -> "
        "CLAHE on L* -> Otsu on b* -> 23 features -> StandardScaler + SVC). The reliability "
        "column is advisory and never changes the predicted class.", small))

    doc.build(story)
    return buf.getvalue()
