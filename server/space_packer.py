"""
Grid-based furniture space packer.

Algorithm
---------
1. Rasterise the room polygon onto a grid of *cell_size* metres.
2. Mark cells occupied by already-placed furniture (AABB of each item).
3. Collect all furniture items to pack (from catalog, filtered by size).
4. Try three placement strategies (left-to-right, right-to-left,
   centre-out) to produce up to *num_strategies* distinct arrangements.
5. For each strategy, sweep the grid and greedily place items whose
   footprint fits in unoccupied cells, respecting a minimum clearance
   from walls and already-placed furniture.

Each strategy returns a list of placed items (world X, Z, rotation_deg,
catalog entry) that can be appended to an existing layout option.
"""
from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

from server.catalog_client import CatalogEntry

logger = logging.getLogger(__name__)

# Grid cell resolution (metres)
_CELL_M = 0.5
# Minimum clearance between furniture edge and wall (metres)
_WALL_CLEARANCE_M = 0.30
# Minimum clearance between two furniture items (metres)
_ITEM_CLEARANCE_M = 0.20

# Candidate rotations to try for each item (degrees)
_ROTATIONS = [0.0, 90.0, 180.0, 270.0]


def _shoelace_area(polygon: list[tuple[float, float]]) -> float:
    """Signed area of a 2-D polygon via shoelace formula."""
    n = len(polygon)
    area = 0.0
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def _point_in_polygon(x: float, z: float, polygon: list[tuple[float, float]]) -> bool:
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


class _Grid:
    """Binary occupancy grid aligned to the room polygon bounding box."""

    def __init__(self, polygon_m: list[tuple[float, float]], cell_m: float) -> None:
        xs = [p[0] for p in polygon_m]
        zs = [p[1] for p in polygon_m]
        self.x_min = min(xs) - cell_m
        self.z_min = min(zs) - cell_m
        self.cell_m = cell_m

        self.nx = int(math.ceil((max(xs) - self.x_min) / cell_m)) + 2
        self.nz = int(math.ceil((max(zs) - self.z_min) / cell_m)) + 2

        # 0=free inside polygon, 1=occupied/wall
        self.cells = np.ones((self.nx, self.nz), dtype=np.uint8)

        # Mark cells inside polygon as free
        for ix in range(self.nx):
            for iz in range(self.nz):
                cx = self.x_min + (ix + 0.5) * cell_m
                cz = self.z_min + (iz + 0.5) * cell_m
                if _point_in_polygon(cx, cz, polygon_m):
                    self.cells[ix, iz] = 0

        # Apply wall clearance: mark cells within _WALL_CLEARANCE_M of boundary as occupied
        wall_cells = int(math.ceil(_WALL_CLEARANCE_M / cell_m))
        free = self.cells == 0
        # Erode free region by wall_cells to keep items away from walls
        for _ in range(wall_cells):
            eroded = free.copy()
            eroded[1:, :] &= free[:-1, :]
            eroded[:-1, :] &= free[1:, :]
            eroded[:, 1:] &= free[:, :-1]
            eroded[:, :-1] &= free[:, 1:]
            free = eroded
        # Mark eroded-away cells as wall-clearance occupied
        self.cells[(self.cells == 0) & ~free] = 2  # value 2 = wall-clearance zone

    def world_to_idx(self, x: float, z: float) -> tuple[int, int]:
        ix = int((x - self.x_min) / self.cell_m)
        iz = int((z - self.z_min) / self.cell_m)
        return ix, iz

    def idx_to_world(self, ix: int, iz: int) -> tuple[float, float]:
        x = self.x_min + (ix + 0.5) * self.cell_m
        z = self.z_min + (iz + 0.5) * self.cell_m
        return x, z

    def is_free(self, ix: int, iz: int) -> bool:
        return (
            0 <= ix < self.nx
            and 0 <= iz < self.nz
            and self.cells[ix, iz] == 0
        )

    def mark_occupied(
        self,
        world_x: float,
        world_z: float,
        half_w: float,
        half_d: float,
        clearance: float = _ITEM_CLEARANCE_M,
    ) -> None:
        """Mark cells covered by an AABB (plus clearance) as occupied."""
        hx = half_w + clearance
        hz = half_d + clearance
        ix0, iz0 = self.world_to_idx(world_x - hx, world_z - hz)
        ix1, iz1 = self.world_to_idx(world_x + hx, world_z + hz)
        for ix in range(max(0, ix0), min(self.nx, ix1 + 1)):
            for iz in range(max(0, iz0), min(self.nz, iz1 + 1)):
                if self.cells[ix, iz] == 0:
                    self.cells[ix, iz] = 1

    def fits(
        self,
        world_x: float,
        world_z: float,
        half_w: float,
        half_d: float,
    ) -> bool:
        """Return True if the AABB footprint is entirely on free cells."""
        ix0, iz0 = self.world_to_idx(world_x - half_w, world_z - half_d)
        ix1, iz1 = self.world_to_idx(world_x + half_w, world_z + half_d)
        if ix0 < 0 or iz0 < 0 or ix1 >= self.nx or iz1 >= self.nz:
            return False
        for ix in range(ix0, ix1 + 1):
            for iz in range(iz0, iz1 + 1):
                if self.cells[ix, iz] != 0:
                    return False
        return True

    def free_area_m2(self) -> float:
        return float(np.sum(self.cells == 0)) * self.cell_m ** 2


def _iter_positions(
    grid: _Grid,
    strategy: int,
) -> list[tuple[int, int]]:
    """
    Return a list of (ix, iz) grid indices in sweep order for the given strategy.

    strategy 0 : row-major, left-to-right / top-to-bottom
    strategy 1 : row-major, right-to-left / top-to-bottom
    strategy 2 : spiral from centre outward
    """
    free_ixs = [
        (ix, iz)
        for ix in range(grid.nx)
        for iz in range(grid.nz)
        if grid.cells[ix, iz] == 0
    ]
    if not free_ixs:
        return []

    if strategy == 0:
        return sorted(free_ixs, key=lambda p: (p[0], p[1]))
    if strategy == 1:
        return sorted(free_ixs, key=lambda p: (-p[0], p[1]))
    # strategy 2: sort by distance from centre of free cells
    cx = sum(p[0] for p in free_ixs) / len(free_ixs)
    cz = sum(p[1] for p in free_ixs) / len(free_ixs)
    return sorted(free_ixs, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cz) ** 2)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_free_area_m2(
    polygon_m: list[tuple[float, float]],
    existing_objects: list[dict],
) -> float:
    """
    Return the estimated free floor area (m²) after subtracting placed furniture.

    ``existing_objects`` is a list of dicts with at least ``position`` and
    ``size`` keys (same format as returned by ``APMRunner.predict``).
    """
    grid = _Grid(polygon_m, _CELL_M)
    for obj in existing_objects:
        pos = obj.get("position_apm") or obj.get("position") or {}
        size = obj.get("size_m") or obj.get("size") or {}
        if isinstance(pos, dict):
            x, z = float(pos.get("x", 0)), float(pos.get("z", 0))
        elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
            x, z = float(pos[0]), float(pos[2])
        else:
            continue
        if isinstance(size, dict):
            w, d = float(size.get("w", 1.0)) / 2, float(size.get("d", 1.0)) / 2
        elif isinstance(size, (list, tuple)) and len(size) >= 3:
            w, d = float(size[0]) / 2, float(size[2]) / 2
        else:
            continue
        grid.mark_occupied(x, z, w, d, clearance=0.0)
    return grid.free_area_m2()


def pack_furniture(
    polygon_m: list[tuple[float, float]],
    placed_objects_world: list[dict],
    catalog_items: list[tuple[str, CatalogEntry]],
    num_strategies: int = 3,
) -> list[list[dict]]:
    """
    Pack *catalog_items* into the free space of the room.

    Parameters
    ----------
    polygon_m : room outline in metres [[x, z], ...]
    placed_objects_world : objects already placed; each dict has at least
        ``position`` (PipelineNormalizeRunPosition-like, .x/.z or list) and
        ``size`` ([w, h, d] or dict with w/h/d).
    catalog_items : list of (catalog_type_str, CatalogEntry) to place
    num_strategies : number of independent sweep strategies to try

    Returns
    -------
    A list (one per strategy) of lists of newly placed object dicts:
        {"catalog_type": str, "entry": CatalogEntry,
         "world_x": float, "world_z": float, "rotation_deg": float}
    """
    if not catalog_items:
        return [[] for _ in range(num_strategies)]

    results: list[list[dict]] = []

    for strategy in range(num_strategies):
        # Fresh grid for each strategy
        grid = _Grid(polygon_m, _CELL_M)

        # Mark cells occupied by already-placed items
        for obj in placed_objects_world:
            pos = obj.get("position")
            size = obj.get("size")
            if pos is None or size is None:
                continue
            if hasattr(pos, "x"):
                wx, wz = pos.x, pos.z
            elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
                wx, wz = float(pos[0]), float(pos[2])
            elif isinstance(pos, dict):
                wx, wz = float(pos.get("x", 0)), float(pos.get("z", 0))
            else:
                continue
            if isinstance(size, (list, tuple)) and len(size) >= 3:
                hw, hd = float(size[0]) / 2, float(size[2]) / 2
            elif isinstance(size, dict):
                hw, hd = float(size.get("w", 1)) / 2, float(size.get("d", 1)) / 2
            else:
                continue
            grid.mark_occupied(wx, wz, hw, hd)

        sweep_order = _iter_positions(grid, strategy)
        placed: list[dict] = []

        for cat_type, entry in catalog_items:
            if entry.size_m is None:
                continue
            item_w, item_h, item_d = entry.size_m  # (width, height, depth) in metres
            placed_this = False

            for rot_deg in _ROTATIONS:
                rad = math.radians(rot_deg)
                cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
                # Rotated AABB half-extents
                half_w = (item_w * cos_a + item_d * sin_a) / 2.0
                half_d = (item_w * sin_a + item_d * cos_a) / 2.0

                for ix, iz in sweep_order:
                    if not grid.is_free(ix, iz):
                        continue
                    world_x, world_z = grid.idx_to_world(ix, iz)
                    if grid.fits(world_x, world_z, half_w, half_d):
                        # Place it
                        grid.mark_occupied(world_x, world_z, half_w, half_d)
                        placed.append({
                            "catalog_type": cat_type,
                            "entry": entry,
                            "world_x": world_x,
                            "world_z": world_z,
                            "rotation_deg": rot_deg,
                        })
                        placed_this = True
                        break
                if placed_this:
                    break

            if not placed_this:
                logger.debug(
                    "pack_furniture(strategy=%d): no space for %r (%.2f×%.2f m)",
                    strategy, cat_type, item_w, item_d,
                )

        results.append(placed)

    return results
