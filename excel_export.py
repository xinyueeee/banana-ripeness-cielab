"""
excel_export.py  --  write a results table to .xlsx with an embedded thumbnail per
row (Extra Effort: automated export of results/findings; xlsx keeps the image).

A plain .csv cannot hold images; this produces a real Excel workbook where the
first column shows a small picture of each analysed banana next to its result.
"""
from __future__ import annotations

import io

from PIL import Image
import xlsxwriter

_THUMB_PX = 84          # embedded image size
_ROW_PX = 92            # row height so the image fits with a little padding


def _thumb_png(data: bytes, px: int = _THUMB_PX) -> io.BytesIO | None:
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((px, px))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except Exception:  # noqa: BLE001
        return None


def build_xlsx(rows: list[dict], images: dict[str, bytes], *,
               sheet_name: str = "Results", image_key: str = "image") -> bytes:
    """rows: list of dicts (all with the same keys). images: {row[image_key]: bytes}."""
    out = io.BytesIO()
    wb = xlsxwriter.Workbook(out, {"in_memory": True})
    ws = wb.add_worksheet(sheet_name[:31])

    header_fmt = wb.add_format({"bold": True, "bg_color": "#FFE066", "font_color": "#4A3220",
                                "border": 1, "align": "center", "valign": "vcenter"})
    cell_fmt = wb.add_format({"valign": "vcenter", "border": 1})
    img_cell_fmt = wb.add_format({"valign": "vcenter", "align": "center", "border": 1})

    data_keys = [k for k in (rows[0].keys() if rows else [])]
    ws.write(0, 0, "preview", header_fmt)
    for c, key in enumerate(data_keys, start=1):
        ws.write(0, c, key, header_fmt)

    ws.set_column(0, 0, 14)                      # preview column
    ws.set_column(1, len(data_keys), 16)

    for r, row in enumerate(rows, start=1):
        ws.set_row(r, _ROW_PX * 0.75)            # points
        buf = _thumb_png(images.get(row.get(image_key, ""), b""))
        if buf is not None:
            ws.write(r, 0, "", img_cell_fmt)
            ws.insert_image(r, 0, "t.png", {
                "image_data": buf, "object_position": 1,
                "x_offset": 6, "y_offset": 4,
            })
        else:
            ws.write(r, 0, "n/a", img_cell_fmt)
        for c, key in enumerate(data_keys, start=1):
            v = row[key]
            ws.write(r, c, "" if v is None else v, cell_fmt)

    ws.freeze_panes(1, 0)
    wb.close()
    return out.getvalue()
