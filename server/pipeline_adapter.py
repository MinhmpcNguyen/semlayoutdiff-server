"""
Main pipeline adapter.

Bridges PipelineNormalizeRunRequest → backend_new ML pipeline →
PipelineNormalizeRunResponse, bridging the gaps between:
  - Frontend polygon input  ←→  SLDN binary floor plan image
  - APM furniture predictions  ←→  catalog item lookup for modelUrl/catalogItemId
  - APM coordinate space  ←→  frontend world coordinate space
"""
from __future__ import annotations

import json
import logging
import os
from typing import Sequence

import numpy as np

from server.catalog_client import CatalogEntry, CatalogIndex
from server.domain import (
    FrontendOpeningPayload,
    PipelineNormalizeRunDebugZone,
    PipelineNormalizeRunObject,
    PipelineNormalizeRunOption,
    PipelineNormalizeRunPosition,
    PipelineNormalizeRunRequest,
    PipelineNormalizeRunResponse,
    PipelineNormalizeRunRotation,
)
from server.floor_plan_utils import (
    _to_meters,
    apm_position_to_world,
    detect_room_type,
    polygon_to_floor_plan_tensor,
    rotation_deg_to_quaternion,
)
from server.ml_runner import APMRunner, SLDNRunner

logger = logging.getLogger(__name__)


def _point_in_polygon(x: float, z: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test on the X-Z floor plane."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, zi = polygon[i]
        xj, zj = polygon[j]
        if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
            inside = not inside
        j = i
    return inside


def _save_debug_files(
    floor_plan: np.ndarray,
    semantic_maps: list[np.ndarray],
    room_center: tuple[float, float],
    scale_px: float,
) -> None:
    """Save intermediate SLDN inputs/outputs to SLDN_DEBUG_DIR if the env var is set."""
    debug_dir = os.environ.get("SLDN_DEBUG_DIR", "").strip()
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    np.save(os.path.join(debug_dir, "floor_plan.npy"), floor_plan)
    for i, sem in enumerate(semantic_maps):
        np.save(os.path.join(debug_dir, f"semantic_map_{i}.npy"), sem)
    with open(os.path.join(debug_dir, "meta.json"), "w") as fh:
        json.dump(
            {
                "center": list(room_center),
                "scale_px": scale_px,
                "num_options": len(semantic_maps),
            },
            fh,
        )
    logger.info("Debug files saved to %s (%d semantic maps)", debug_dir, len(semantic_maps))


# Placeholder model URL used when no catalog item is found.
_FALLBACK_MODEL_URL = "https://storage.mazig.io/models/placeholder.glb"

# Categories that hang from the ceiling — their Y position must NOT be zeroed out.
_CEILING_CATEGORIES: frozenset[str] = frozenset({
    "ceiling_light", "pendant_lamp", "ceiling_lamp",
})

# Room area (m²) → coverage_ratio heuristic:
# assume a good layout covers ~30-50 % of the floor area with furniture.
_TARGET_COVERAGE_RATIO = 0.40

# Minimum clearance (metres) between furniture centroid and an actual door/window.
# SLDN has no knowledge of real opening positions, so we post-filter any item
# whose centroid falls within this radius of a door or window from req.openings.
_OPENING_CLEARANCE_M = 1.0


class PipelineAdapter:
    """
    Orchestrates the full request → response conversion using ML models.

    Parameters
    ----------
    sldn_runner : SLDNRunner
        Loaded SLDN model for semantic map generation.
    apm_runner : APMRunner
        Loaded APM model for attribute prediction.
    catalog_index : CatalogIndex
        Pre-loaded catalog index for model URL lookups.
    num_options : int
        Number of independent layout samples to generate (= number of options
        returned to the frontend).  Default 3.
    """

    def __init__(
        self,
        sldn_runner: SLDNRunner,
        apm_runner: APMRunner,
        catalog_index: CatalogIndex,
        num_options: int = 3,
    ) -> None:
        self._sldn = sldn_runner
        self._apm = apm_runner
        self._catalog = catalog_index
        self._num_options = num_options

    def run(self, req: PipelineNormalizeRunRequest) -> PipelineNormalizeRunResponse:
        polygons = req.room.polygons
        if not polygons or len(polygons) < 3:
            raise ValueError("Room polygon must have at least 3 vertices.")

        # ── Step 1: Convert polygon → floor plan tensor ───────────────────────
        floor_plan_tensor, room_center, scale_px = polygon_to_floor_plan_tensor(
            polygons, req.source_unit
        )
        room_polygon_m = _to_meters(polygons, req.source_unit)
        room_type_id = detect_room_type(req.room.name, req.room.description)
        room_type_str = {0: "bedroom", 1: "livingroom", 2: "diningroom"}.get(
            room_type_id, "livingroom"
        )

        # ── Step 2: SLDN → semantic maps (one per option) ────────────────────
        try:
            semantic_maps = self._sldn.generate(
                floor_plan_tensor, room_type_id, num_samples=self._num_options
            )
        except Exception as exc:
            raise RuntimeError(f"SLDN inference failed: {exc}") from exc

        _save_debug_files(
            floor_plan_tensor.squeeze().cpu().numpy(),
            semantic_maps,
            room_center,
            scale_px,
        )

        # ── Step 3: APM → furniture attributes (per option) ──────────────────
        option_furniture_lists: list[list[dict]] = []
        for sem_map in semantic_maps:
            try:
                furniture = self._apm.predict(sem_map)
            except Exception as exc:
                logger.warning("APM inference failed for one option: %s", exc)
                furniture = []
            option_furniture_lists.append(furniture)

        # ── Step 3b: Extract real door/window positions from req.openings ────
        # FrontendOpeningPayload stores extra fields (position, size, …) via
        # Pydantic's extra="allow" — the frontend sends full SceneObject JSON.
        # We extract (x, z) world positions so we can clear furniture near them.
        opening_positions: list[tuple[float, float]] = []
        for op in req.openings:
            extra = op.model_extra or {}
            pos = extra.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                opening_positions.append((float(pos[0]), float(pos[2])))
            elif isinstance(pos, dict):
                opening_positions.append((float(pos.get("x", 0)), float(pos.get("z", 0))))
        if opening_positions:
            logger.info(
                "Door/window clearance active: %d opening(s) at %s",
                len(opening_positions),
                [(round(x, 2), round(z, 2)) for x, z in opening_positions],
            )

        # ── Step 4: Build PipelineNormalizeRunOption objects ─────────────────
        options: list[PipelineNormalizeRunOption] = []
        best_option_id: str | None = None
        best_score: int = -1

        for idx, furniture_list in enumerate(option_furniture_lists):
            option_id = f"option_{idx + 1}"
            objects = self._furniture_to_objects(
                furniture_list, room_center, scale_px, room_polygon_m,
                opening_positions=opening_positions or None,
            )
            score = len(objects)  # simple score: more furniture = better layout
            coverage = self._coverage_ratio(furniture_list, polygons, req.source_unit)

            option = PipelineNormalizeRunOption(
                optionId=option_id,
                label=f"Layout {idx + 1}",
                layoutScore=score,
                hardValid=score > 0,
                complete=score > 0,
                coverageRatio=round(coverage, 3),
                objects=objects,
                openings=list(req.openings),
            )
            options.append(option)

            if score > best_score:
                best_score = score
                best_option_id = option_id

        if not options:
            return PipelineNormalizeRunResponse(
                objects=[],
                openings=list(req.openings),
                selectedOptionId=None,
                options=[],
                debugZones=[
                    PipelineNormalizeRunDebugZone(
                        roomId="room_1",
                        roomType=room_type_str,
                        polygon=[(p[0], p[1]) for p in polygons],
                    )
                ],
            )

        # ── Step 5: Assemble response ─────────────────────────────────────────
        selected = next(
            (o for o in options if o.optionId == best_option_id), options[0]
        )

        return PipelineNormalizeRunResponse(
            objects=selected.objects,
            openings=list(req.openings),
            selectedOptionId=best_option_id,
            options=options,
            debugZones=[
                PipelineNormalizeRunDebugZone(
                    roomId="room_1",
                    roomType=room_type_str,
                    polygon=[(p[0], p[1]) for p in polygons],
                )
            ],
        )

    # ── Conversion helpers ────────────────────────────────────────────────────

    def _furniture_to_objects(
        self,
        furniture_list: list[dict],
        room_center: tuple[float, float],
        scale_px: float,
        room_polygon_m: list[tuple[float, float]],
        opening_positions: list[tuple[float, float]] | None = None,
    ) -> list[PipelineNormalizeRunObject]:
        objects: list[PipelineNormalizeRunObject] = []
        for item in furniture_list:
            obj = self._furniture_item_to_object(
                item, room_center, scale_px, room_polygon_m,
                opening_positions=opening_positions,
            )
            if obj is not None:
                objects.append(obj)
        return objects

    def _furniture_item_to_object(
        self,
        item: dict,
        room_center: tuple[float, float],
        scale_px: float,
        room_polygon_m: list[tuple[float, float]],
        opening_positions: list[tuple[float, float]] | None = None,
    ) -> PipelineNormalizeRunObject | None:
        category = item.get("category", "unknown")
        pos_apm = item.get("position_apm", {})
        size_apm = item.get("size_m", {})
        rotation_deg = float(item.get("rotation_deg", 0.0))

        # Catalog lookup
        catalog_entry: CatalogEntry | None = self._catalog.lookup(category)

        # Position: convert from APM space to world coordinates
        world_x, world_y, world_z = apm_position_to_world(
            x_3d=float(pos_apm.get("x", 0.0)),
            y_3d=float(pos_apm.get("y", 0.0)),
            z_3d=float(pos_apm.get("z", 0.0)),
            room_center=room_center,
            scale_px=scale_px,
        )
        # Fix 1 — Y position: APM offset_pred is very noisy (loss weight=0.01).
        # Floor items must sit on the floor (Y=0); only ceiling items keep APM's Y.
        if category in _CEILING_CATEGORIES:
            world_y = max(0.0, world_y)
        else:
            world_y = 0.0

        # Discard items whose centroid falls outside the room polygon
        if room_polygon_m and not _point_in_polygon(world_x, world_z, room_polygon_m):
            logger.debug(
                "Skipping out-of-bounds %r at (%.3f, %.3f)", category, world_x, world_z
            )
            return None

        # Rotation: degrees → quaternion
        qx, qy, qz, qw = rotation_deg_to_quaternion(rotation_deg)

        # Prefer catalog size; fall back to APM prediction
        if catalog_entry and catalog_entry.size_m:
            size = list(catalog_entry.size_m)
        else:
            w = float(size_apm.get("w", 1.0))
            h = float(size_apm.get("h", 0.5))
            d = float(size_apm.get("d", 1.0))
            size = [w, h, d]

        # Fix 2 — Footprint clamping: centroid may be inside the polygon but the
        # full furniture footprint (catalog size × rotation) can still overflow the walls.
        # Compute the axis-aligned bounding box of the rotated footprint and clamp
        # the centroid so every corner stays within the room bounding box.
        if room_polygon_m:
            import math as _math
            half_w, half_d = size[0] / 2.0, size[2] / 2.0
            rad = _math.radians(rotation_deg)
            cos_a, sin_a = abs(_math.cos(rad)), abs(_math.sin(rad))
            # AABB half-extents of the rotated footprint in world X and Z
            hx = half_w * cos_a + half_d * sin_a
            hz = half_w * sin_a + half_d * cos_a
            # Shrink room bounding box by those half-extents
            xs = [p[0] for p in room_polygon_m]
            zs = [p[1] for p in room_polygon_m]
            lo_x, hi_x = min(xs) + hx, max(xs) - hx
            lo_z, hi_z = min(zs) + hz, max(zs) - hz
            if lo_x > hi_x or lo_z > hi_z:
                logger.debug(
                    "Room too small for %r (%.2f×%.2fm) — skipping.", category, size[0], size[2]
                )
                return None
            world_x = max(lo_x, min(world_x, hi_x))
            world_z = max(lo_z, min(world_z, hi_z))

        # Fix 3 — Door/window clearance: skip furniture whose centroid is within
        # _OPENING_CLEARANCE_M of any actual door or window position.
        # SLDN does not receive real opening locations so it may place furniture
        # in front of actual doors.  We post-filter using req.openings positions.
        if opening_positions:
            for ox, oz in opening_positions:
                dist = ((world_x - ox) ** 2 + (world_z - oz) ** 2) ** 0.5
                if dist < _OPENING_CLEARANCE_M:
                    logger.debug(
                        "Skipping %r at (%.3f, %.3f): within %.2fm of opening at (%.3f, %.3f)",
                        category, world_x, world_z, dist, ox, oz,
                    )
                    return None

        model_url = (
            catalog_entry.model_url if catalog_entry else _FALLBACK_MODEL_URL
        )
        catalog_item_id = (
            catalog_entry.catalog_item_id if catalog_entry else None
        )
        name = catalog_entry.name if catalog_entry else category
        color = catalog_entry.color if catalog_entry else None
        object_role = catalog_entry.object_role if catalog_entry else None

        if model_url == _FALLBACK_MODEL_URL and catalog_item_id is None:
            # Skip items with no model at all
            logger.debug("No catalog entry for category %r, skipping.", category)
            return None

        return PipelineNormalizeRunObject(
            name=name,
            size=size,
            type="model",
            color=color,
            modelUrl=model_url,
            position=PipelineNormalizeRunPosition(x=world_x, y=world_y, z=world_z),
            rotation=PipelineNormalizeRunRotation(x=qx, y=qy, z=qz, w=qw),
            objectRole=object_role,
            catalogItemId=catalog_item_id,
            collisionLayer="floor_solid",
        )

    @staticmethod
    def _coverage_ratio(
        furniture_list: list[dict],
        polygons: Sequence,
        source_unit: str,
    ) -> float:
        """Rough coverage: sum of furniture footprints / room area."""
        if not furniture_list:
            return 0.0

        # Estimate room area from polygon bounding box
        pts_m = _to_meters(polygons, source_unit)
        xs = [p[0] for p in pts_m]
        ys = [p[1] for p in pts_m]
        room_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if room_area <= 0:
            return 0.0

        furniture_area = sum(
            item["size_m"].get("w", 1.0) * item["size_m"].get("d", 1.0)
            for item in furniture_list
            if "size_m" in item
        )
        return min(furniture_area / room_area, 1.0)
