"""
detection.py  --  object detection overlay, geometric rectification and
object-property measurement.

Everything here is derived from the *final banana mask that the frozen CIELAB
segmentation already produced* (Otsu on b* -> morphology -> largest connected
component -> bounded hole-fill). It rectifies that region to a canonical
horizontal pose, draws its bounding box + contour, and measures its pixel
geometry.

This is a MEASUREMENT / DISPLAY layer only. It adds no new segmentation, no
feature fed to the classifier, and no colour space beyond CIELAB -- the 23
classifier features are still computed on the un-rectified frozen pipeline, so
the reported accuracy is unchanged. Its job is the Shared Core "Image
Calibration" and "Object Detection" requirements.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from config import IMAGE_SIZE, CANVAS_PX


@dataclass
class ObjectProperties:
    detected: bool
    rotation_deg: float                 # angle applied to level the banana's principal axis
    bbox_x: int
    bbox_y: int
    bbox_w: int                         # axis-aligned bbox on the RECTIFIED mask
    bbox_h: int
    area_px: int
    frame_fraction_pct: float
    bbox_fill_pct: float                # banana px / bbox px  (extent, on rectified mask)
    aspect_ratio: float                 # length / width  (pose-independent)
    length_px: float
    width_px: float
    contour_perimeter_px: float
    centroid_x: int
    centroid_y: int

    def as_table_rows(self) -> list[tuple[str, str]]:
        if not self.detected:
            return [("Object detected", "no banana region found")]
        return [
            ("Rectification angle", f"{self.rotation_deg:+.1f} deg (principal axis -> horizontal)"),
            ("Bounding box on rectified image (x, y, w, h)",
             f"({self.bbox_x}, {self.bbox_y}, {self.bbox_w}, {self.bbox_h}) px"),
            ("Banana area", f"{self.area_px:,} px  ({self.frame_fraction_pct:.1f}% of frame)"),
            ("Length x width", f"{self.length_px:.0f} x {self.width_px:.0f} px"),
            ("Aspect ratio (length / width)", f"{self.aspect_ratio:.2f}"),
            ("Bounding-box fill (extent)", f"{self.bbox_fill_pct:.1f}%"),
            ("Contour perimeter", f"{self.contour_perimeter_px:.0f} px"),
            ("Centroid (x, y)", f"({self.centroid_x}, {self.centroid_y}) px"),
        ]

    def to_dict(self) -> dict:
        return asdict(self)


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _principal_axis_angle(cnt) -> float:
    """Angle (degrees) of the banana's long axis, from the min-area rectangle's
    longest edge. Robust across OpenCV versions (does not rely on minAreaRect's
    angle convention)."""
    box = cv2.boxPoints(cv2.minAreaRect(cnt))
    edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    p1, p2 = max(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))
    ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    # fold into (-90, 90] so we rotate by the smallest amount
    if ang > 90:
        ang -= 180
    elif ang <= -90:
        ang += 180
    return float(ang)


def rectify(rgb_resized: np.ndarray, mask: np.ndarray):
    """Rotate image + mask so the banana's principal axis is horizontal.

    Returns (rectified_rgb, rectified_mask uint8 0/255, angle_deg). Pure geometry:
    the classifier never sees this -- it only makes the axis-aligned bounding box,
    aspect ratio and extent pose-independent for the calibration read-out.
    """
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    cnt = _largest_contour(binary)
    if cnt is None:
        return np.ascontiguousarray(rgb_resized), binary, 0.0
    angle = _principal_axis_angle(cnt)
    h, w = binary.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rgb_r = cv2.warpAffine(rgb_resized, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderValue=(255, 255, 255))
    mask_r = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST)
    mask_r = np.where(mask_r > 127, 255, 0).astype(np.uint8)
    return np.ascontiguousarray(rgb_r), mask_r, angle


def measure(mask: np.ndarray) -> ObjectProperties:
    """Rectify the banana to a horizontal pose, then measure it. Input is the
    frozen segmentation mask (0/255 uint8, 416x416)."""
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0 or _largest_contour(binary) is None:
        return ObjectProperties(False, 0.0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    _, rect_mask, angle = rectify(np.zeros((*binary.shape, 3), np.uint8), binary)
    cnt = _largest_contour(rect_mask)
    if cnt is None:                      # rotation edge case -> fall back to raw
        rect_mask, cnt, angle = binary, _largest_contour(binary), 0.0

    area = int(np.count_nonzero(rect_mask))
    x, y, w, h = cv2.boundingRect(cnt)
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    length_px, width_px = (max(rw, rh), max(min(rw, rh), 1.0))
    m = cv2.moments(rect_mask, binaryImage=True)
    cx = int(m["m10"] / m["m00"]) if m["m00"] else x + w // 2
    cy = int(m["m01"] / m["m00"]) if m["m00"] else y + h // 2

    return ObjectProperties(
        detected=True, rotation_deg=float(angle),
        bbox_x=int(x), bbox_y=int(y), bbox_w=int(w), bbox_h=int(h),
        area_px=area,
        frame_fraction_pct=100.0 * area / CANVAS_PX,
        bbox_fill_pct=100.0 * area / float(max(w * h, 1)),
        aspect_ratio=float(length_px) / float(width_px),
        length_px=float(length_px), width_px=float(width_px),
        contour_perimeter_px=float(cv2.arcLength(cnt, True)),
        centroid_x=cx, centroid_y=cy,
    )


def detection_overlay(rgb_resized: np.ndarray, mask: np.ndarray,
                      label: str | None = None) -> np.ndarray:
    """416x416 RGB with the banana contour (green) + bounding box (blue) + label."""
    out = np.ascontiguousarray(rgb_resized.copy())
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    cnt = _largest_contour(binary)
    if cnt is None:
        return out
    cv2.drawContours(out, [cnt], -1, (46, 204, 113), 2)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(out, (x, y), (x + w, y + h), (52, 152, 219), 2)
    if label:
        tag = label.upper()
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y - 8, th + 6)
        cv2.rectangle(out, (x, ty - th - 6), (x + tw + 8, ty + 2), (52, 152, 219), -1)
        cv2.putText(out, tag, (x + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out


def rectified_overlay(rgb_resized: np.ndarray, mask: np.ndarray,
                      label: str | None = None) -> tuple[np.ndarray, float]:
    """Rectified image with the bounding box drawn on the levelled banana."""
    rgb_r, mask_r, angle = rectify(rgb_resized, mask)
    return detection_overlay(rgb_r, mask_r, label), angle


def calibration_rows(rotation_deg: float | None = None) -> list[tuple[str, str]]:
    rows = [
        ("Spatial scaling", f"resize to {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} px "
                            f"(cv2.INTER_AREA) -- equal pixel budget for every image"),
        ("Geometric rectification", "banana principal axis rotated to horizontal "
                                    "(from the min-area rectangle) before measurement"),
        ("Consistency", "bounding box, aspect ratio and extent are pose-independent "
                        "and comparable between images"),
        ("Physical (mm) calibration", "not applicable -- the dataset provides no scale "
                                      "reference; measurements are reported in pixels"),
    ]
    if rotation_deg is not None:
        rows.insert(2, ("Rectification applied here", f"{rotation_deg:+.1f} deg"))
    return rows
