"""
Main pipeline adapter.

Bridges PipelineNormalizeRunRequest → backend_new ML pipeline →
PipelineNormalizeRunResponse, bridging the gaps between:
  - Frontend polygon input  ←→  SLDN binary floor plan image
  - APM furniture predictions  ←→  catalog item lookup for modelUrl/catalogItemId
  - APM coordinate space  ←→  frontend world coordinate space
"""
from __future__ import annotations

import logging
from typing import Sequence

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
    apm_position_to_world,
    detect_room_type,
    polygon_to_floor_plan_tensor,
    rotation_deg_to_quaternion,
)
from server.ml_runner import APMRunner, SLDNRunner

logger = logging.getLogger(__name__)

# Placeholder model URL used when no catalog item is found.
_FALLBACK_MODEL_URL = "https://storage.mazig.io/models/placeholder.glb"

# Room area (m²) → coverage_ratio heuristic:
# assume a good layout covers ~30-50 % of the floor area with furniture.
_TARGET_COVERAGE_RATIO = 0.40


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

        # ── Step 3: APM → furniture attributes (per option) ──────────────────
        option_furniture_lists: list[list[dict]] = []
        for sem_map in semantic_maps:
            try:
                furniture = self._apm.predict(sem_map)
            except Exception as exc:
                logger.warning("APM inference failed for one option: %s", exc)
                furniture = []
            option_furniture_lists.append(furniture)

        # ── Step 4: Build PipelineNormalizeRunOption objects ─────────────────
        options: list[PipelineNormalizeRunOption] = []
        best_option_id: str | None = None
        best_score: int = -1

        for idx, furniture_list in enumerate(option_furniture_lists):
            option_id = f"option_{idx + 1}"
            objects = self._furniture_to_objects(
                furniture_list, room_center, scale_px
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
    ) -> list[PipelineNormalizeRunObject]:
        objects: list[PipelineNormalizeRunObject] = []
        for item in furniture_list:
            obj = self._furniture_item_to_object(item, room_center, scale_px)
            if obj is not None:
                objects.append(obj)
        return objects

    def _furniture_item_to_object(
        self,
        item: dict,
        room_center: tuple[float, float],
        scale_px: float,
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
        from server.floor_plan_utils import _to_meters  # noqa: PLC0415

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
