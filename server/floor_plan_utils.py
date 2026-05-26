"""
Utilities for converting frontend room polygon → SLDN floor plan image,
and for detecting room type from room name/description.

Coordinate system:
  - Frontend polygon: [[x, y], ...] in 2D (x=right, y=forward)
  - SLDN image (120×120): row = 60 + (polygon_x - cx) * scale_px
                           col = 60 + (polygon_y - cy) * scale_px
  - APM output (1200×1200 upscale): x_3d from rows, z_3d from cols
  - With scale_px = 10 px/m the APM positions are already in meters
    centered on the room center, so:
      frontend_x = apm_x_3d + cx
      frontend_z = apm_z_3d + cy
  - For rooms wider than 12 m the scale is reduced; a correction factor
    is applied when converting APM positions back to real-world coords.
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
import torch

# SLDN model operates on 120×120 binary images (1 px = 0.1 m at default scale)
SLDN_IMAGE_SIZE = 120
# At 10 px/m, the image represents a 12 m × 12 m room
_DEFAULT_SCALE_PX_PER_M = 10.0
_IMAGE_CENTER = SLDN_IMAGE_SIZE / 2  # 60.0

# Unit multipliers to convert to metres
_UNIT_SCALE: dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
}

# ── Room type detection ───────────────────────────────────────────────────────

_BEDROOM_KEYWORDS = {
    "bed", "bedroom", "phòng ngủ", "phong ngu", "sleeping", "master", "single",
    "double", "twin", "giuong", "ngu",
}
_DININGROOM_KEYWORDS = {
    "dining", "diningroom", "phòng ăn", "phong an", "eat", "ăn", "an",
    "kitchen", "bếp", "bep",
}
_LIVINGROOM_KEYWORDS = {
    "living", "livingroom", "phòng khách", "phong khach", "lounge", "sitting",
    "family", "khách", "khach", "sofa", "tv",
}

# APM room_type integer: 0=bedroom, 1=livingroom, 2=diningroom
ROOM_TYPE_BEDROOM = 0
ROOM_TYPE_LIVINGROOM = 1
ROOM_TYPE_DININGROOM = 2


def detect_room_type(name: str | None, description: str | None) -> int:
    """Return APM room_type integer (0/1/2) from room name/description."""
    text = " ".join(filter(None, [name, description])).lower()
    if not text:
        return ROOM_TYPE_LIVINGROOM  # default fallback

    if any(kw in text for kw in _BEDROOM_KEYWORDS):
        return ROOM_TYPE_BEDROOM
    if any(kw in text for kw in _DININGROOM_KEYWORDS):
        return ROOM_TYPE_DININGROOM
    if any(kw in text for kw in _LIVINGROOM_KEYWORDS):
        return ROOM_TYPE_LIVINGROOM
    return ROOM_TYPE_LIVINGROOM


# ── Floor plan rasterization ──────────────────────────────────────────────────

def polygon_to_floor_plan_tensor(
    polygons: Sequence[Sequence[float]],
    source_unit: str = "auto",
    openings: list[dict] | None = None,
) -> tuple[torch.Tensor, tuple[float, float], float]:
    """
    Convert a 2D room polygon to a 120×120 floor plan tensor.

    When *openings* is provided the tensor uses the "arch" encoding:
      0 = background, 1 = floor, 2 = door, 3 = window
    Without openings it produces a plain binary tensor (0/1).

    Parameters
    ----------
    polygons : list of [x, z] points describing the room boundary
    source_unit : unit of polygon coordinates ("m", "mm", "auto")
    openings : optional list of dicts, each with keys:
        "objectRole" : "door" | "window"
        "position"   : [x, y, z]  (world coords, same unit as polygons)
        "size"       : [w, h, d]  (world units, optional)
        "rotation"   : [qx, qy, qz, qw]  (optional)

    Returns
    -------
    tensor : torch.Tensor  shape [1, 1, 120, 120], dtype int64
    room_center : (cx, cy)  room centre in metres
    scale_px : float  pixels-per-metre
    """
    pts_m = _to_meters(polygons, source_unit)

    xs = [p[0] for p in pts_m]
    ys = [p[1] for p in pts_m]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)

    room_size = max(w, h)
    if room_size <= 0:
        room_size = 1.0

    # Fit room into 100 px (leaving 10 px margin each side in the 120 px image)
    # Cap at the default scale so APM positions stay in the correct metric range
    scale_px = min(_DEFAULT_SCALE_PX_PER_M, 100.0 / room_size)

    pixel_pts = np.array(
        [
            [
                int(round(_IMAGE_CENTER + (x - cx) * scale_px)),
                int(round(_IMAGE_CENTER + (y - cy) * scale_px)),
            ]
            for x, y in pts_m
        ],
        dtype=np.int32,
    )

    img = np.zeros((SLDN_IMAGE_SIZE, SLDN_IMAGE_SIZE), dtype=np.uint8)
    # Note: cv2.fillPoly expects (col, row) ordering for points
    # Our pixel_pts are (row_offset, col_offset) relative to center.
    # SLDN row = 60 + (polygon_x - cx) * scale_px → swap axes for cv2
    pts_cv2 = pixel_pts[:, [1, 0]]  # swap to (col, row) = cv2 (x, y)
    cv2.fillPoly(img, [pts_cv2], 1)

    # Arch encoding: draw door (2) and window (3) pixels on top of the floor
    if openings:
        _draw_openings_on_img(img, openings, cx, cy, scale_px)

    tensor = torch.tensor(img.astype(np.int64), dtype=torch.int64).unsqueeze(0).unsqueeze(0)
    return tensor, (cx, cy), scale_px


def _draw_openings_on_img(
    img: np.ndarray,
    openings: list[dict],
    cx: float,
    cy: float,
    scale_px: float,
) -> None:
    """
    Draw door (pixel value 2) and window (pixel value 3) rectangles onto *img*
    in-place.  Implements the "arch" conditioning encoding used during SLDN
    training (0=bg, 1=floor, 2=door, 3=window).

    Coordinate mapping (same as polygon rasterization):
      row = IMAGE_CENTER + (world_x - cx) * scale_px
      col = IMAGE_CENTER + (world_z - cy) * scale_px

    The rectangle is drawn axis-aligned using the larger of the opening's
    width/depth as the square side, so thin door meshes still produce a
    clearly visible marker.  A minimum footprint of 0.8 m is enforced.
    """
    for op in openings:
        pos = op.get("position")
        if not (isinstance(pos, (list, tuple)) and len(pos) >= 3):
            continue

        world_x, world_z = float(pos[0]), float(pos[2])
        row_c = _IMAGE_CENTER + (world_x - cx) * scale_px
        col_c = _IMAGE_CENTER + (world_z - cy) * scale_px

        role = str(op.get("objectRole") or "door").lower()
        pixel_val = 2 if role == "door" else 3

        # Opening footprint: use max(w, d) as square side so a thin door mesh
        # still produces a clearly visible marker.  Minimum 0.8 m.
        size = op.get("size")
        if isinstance(size, (list, tuple)) and len(size) >= 3:
            side_m = max(float(size[0]), float(size[2]), 0.8)
        else:
            side_m = 0.9  # standard door width fallback
        half_px = max(side_m / 2.0 * scale_px, 1.0)  # at least 1 pixel each side

        r0 = int(max(0, math.floor(row_c - half_px)))
        r1 = int(min(SLDN_IMAGE_SIZE, math.ceil(row_c + half_px)))
        c0 = int(max(0, math.floor(col_c - half_px)))
        c1 = int(min(SLDN_IMAGE_SIZE, math.ceil(col_c + half_px)))

        if r1 > r0 and c1 > c0:
            img[r0:r1, c0:c1] = pixel_val


# ── APM → frontend coordinate inverse transform ───────────────────────────────

def apm_position_to_world(
    x_3d: float,
    y_3d: float,
    z_3d: float,
    room_center: tuple[float, float],
    scale_px: float,
) -> tuple[float, float, float]:
    """
    Convert APM output position to frontend world coordinates.

    APM uses a 1200×1200 image at 0.01 m/px (12 m × 12 m frame).
    The rasterization used scale_px px/m in the 120×120 image, which becomes
    10 * scale_px px/m in the 1200×1200 image.

    x_3d = (world_x - cx) * scale_px * 10 * 0.01 = (world_x - cx) * scale_px / 10
    → world_x = x_3d * 10 / scale_px + cx

    Parameters
    ----------
    x_3d, y_3d, z_3d : APM output (from convert_2d_to_3d x/y, plus offset_pred)
        x_3d = APM "x" (from row direction in image)
        y_3d = height above floor (offset_pred from APM)
        z_3d = APM "y" (from col direction in image)
    room_center : (cx, cy) in metres
    scale_px : scale used during rasterization
    """
    correction = _DEFAULT_SCALE_PX_PER_M / scale_px  # 1.0 when scale_px == 10
    cx, cy = room_center
    world_x = x_3d * correction + cx
    world_z = z_3d * correction + cy
    world_y = y_3d  # height is not affected by the 2D scale
    return world_x, world_y, world_z


# ── Rotation conversion ───────────────────────────────────────────────────────

def rotation_deg_to_quaternion(rotation_deg: float) -> tuple[float, float, float, float]:
    """
    Convert a Y-axis rotation (degrees, 0/90/180/270) to a unit quaternion (x, y, z, w).
    Convention: Three.js / right-hand rule, rotation around world Y-axis.
    """
    angle_rad = math.radians(rotation_deg)
    half = angle_rad / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_meters(
    polygons: Sequence[Sequence[float]], source_unit: str
) -> list[tuple[float, float]]:
    unit = source_unit.lower().strip()
    if unit == "auto":
        # Heuristic: if any absolute coordinate > 500, assume millimetres
        flat = [abs(v) for pt in polygons for v in pt]
        unit = "mm" if flat and max(flat) > 500 else "m"
    scale = _UNIT_SCALE.get(unit, 1.0)
    return [(float(pt[0]) * scale, float(pt[1]) * scale) for pt in polygons]
