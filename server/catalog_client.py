"""
Catalog API client — fetches furniture items from the external catalog.

Provides a pre-loaded index keyed by semantic category name so the pipeline
adapter can resolve category → (catalogItemId, modelUrl, size, color) without
a per-request network call.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "https://auto-furniture-api2.a-star.group"
DEFAULT_ASSET_BASE_URL = "https://storage.mazig.io"
DEFAULT_PAGE_LIMIT = 500
DEFAULT_TIMEOUT_SECONDS = 20.0

# Maps backend_new APM category names → catalog search type keywords.
# Multiple APM categories can map to the same catalog type.
_APM_CATEGORY_TO_CATALOG_TYPE: dict[str, str] = {
    "kids_bed": "bed",
    "single_bed": "bed",
    "double_bed": "bed",
    "corner_side_table": "side_table",
    "round_end_table": "side_table",
    "coffee_table": "coffee_table",
    "console_table": "tv_console",
    "tv_stand": "tv_console",
    "desk": "desk",
    "dressing_table": "dresser",
    "table": "dining_table",
    "dining_table": "dining_table",
    "stool": "stool",
    "dressing_chair": "chair",
    "dining_chair": "chair",
    "chinese_chair": "chair",
    "armchair": "armchair",
    "chair": "chair",
    "lounge_chair": "armchair",
    "loveseat_sofa": "sofa",
    "lazy_sofa": "sofa",
    "sofa": "sofa",
    "multi_seat_sofa": "sofa",
    "chaise_longue_sofa": "sofa",
    "l_shaped_sofa": "sofa",
    "nightstand": "nightstand",
    "shelf": "bookshelf",
    "bookshelf": "bookshelf",
    "children_cabinet": "cabinet",
    "wine_cabinet": "cabinet",
    "cabinet": "cabinet",
    "wardrobe": "wardrobe",
    "pendant_lamp": "ceiling_light",
    "ceiling_lamp": "ceiling_light",
}


@dataclass(frozen=True)
class CatalogEntry:
    catalog_item_id: str
    model_url: str
    name: str
    size_m: tuple[float, float, float] | None
    color: str | None
    object_role: str | None
    default_rotation: tuple[float, float, float, float] | None


class _CatalogItem(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    name: str
    name_vn: str | None = Field(default=None, alias="nameVn")
    model_url: str | None = Field(default=None, alias="modelUrl")
    placement_type: str | None = Field(default=None, alias="placementType")
    size: tuple[float, float, float] | None = None
    color_default: str | None = Field(default=None, alias="colorDefault")
    object_role: str | None = Field(default=None, alias="objectRole")
    default_rotation: tuple[float, float, float, float] | None = Field(
        default=None, alias="defaultRotation"
    )

    @field_validator("size", mode="before")
    @classmethod
    def _clean_size(cls, v: object) -> tuple[float, float, float] | None:
        if not isinstance(v, list | tuple) or len(v) != 3:
            return None
        try:
            s = [float(x) for x in v]
        except (TypeError, ValueError):
            return None
        return (s[0], s[1], s[2]) if all(x > 0 for x in s) else None

    @field_validator("default_rotation", mode="before")
    @classmethod
    def _clean_rotation(cls, v: object) -> tuple[float, float, float, float] | None:
        if v is None or not isinstance(v, list | tuple) or len(v) != 4:
            return None
        try:
            r = [float(x) for x in v]
        except (TypeError, ValueError):
            return None
        return (r[0], r[1], r[2], r[3])

    def to_entry(self, asset_base_url: str) -> CatalogEntry | None:
        raw_url = self.model_url
        if not raw_url:
            return None
        model_url = (
            raw_url
            if raw_url.startswith(("http://", "https://"))
            else urljoin(asset_base_url.rstrip("/") + "/", raw_url.lstrip("/"))
        )
        return CatalogEntry(
            catalog_item_id=self.id,
            model_url=model_url,
            name=self.name_vn or self.name,
            size_m=self.size,
            color=self.color_default,
            object_role=self.object_role,
            default_rotation=self.default_rotation,
        )


class CatalogIndex:
    """Pre-loaded catalog index keyed by APM category name."""

    def __init__(
        self,
        by_type: dict[str, list[CatalogEntry]],
    ) -> None:
        self._by_type = by_type

    def lookup(self, apm_category: str) -> CatalogEntry | None:
        catalog_type = _APM_CATEGORY_TO_CATALOG_TYPE.get(apm_category)
        if catalog_type is None:
            logger.debug("No catalog mapping for APM category %r", apm_category)
            return None
        entries = self._by_type.get(catalog_type)
        if not entries:
            logger.debug("No catalog entries for type %r", catalog_type)
            return None
        return entries[0]

    @classmethod
    def empty(cls) -> "CatalogIndex":
        return cls(by_type={})


@dataclass
class CatalogClientSettings:
    api_base_url: str = DEFAULT_API_BASE_URL
    asset_base_url: str = DEFAULT_ASSET_BASE_URL
    page_limit: int = DEFAULT_PAGE_LIMIT
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def load_catalog_index(settings: CatalogClientSettings | None = None) -> CatalogIndex:
    """Fetch all catalog items and build a CatalogIndex keyed by type."""
    cfg = settings or _settings_from_env()
    client = httpx.Client(timeout=cfg.timeout_seconds)
    try:
        # Fetch all items (paginate)
        all_items: list[_CatalogItem] = []
        offset = 0
        limit = min(cfg.page_limit, DEFAULT_PAGE_LIMIT)
        for _ in range(30):
            resp = client.get(
                urljoin(cfg.api_base_url.rstrip("/") + "/", "api/catalog/items"),
                params={
                    "limit": str(limit),
                    "offset": str(offset),
                    "defaultRotationPresence": "present",
                },
            )
            resp.raise_for_status()
            page = resp.json()
            items_raw = page.get("items", [])
            for raw in items_raw:
                try:
                    item = _CatalogItem.model_validate(raw)
                    if item.default_rotation is not None:
                        all_items.append(item)
                except Exception:
                    pass
            total = page.get("total", 0)
            offset += limit
            if offset >= total:
                break

        logger.info("Loaded %d catalog items from API", len(all_items))
    except Exception as exc:
        logger.warning("Failed to load catalog: %s — using empty catalog", exc)
        return CatalogIndex.empty()
    finally:
        client.close()

    # Build index by catalog type keyword
    by_type: dict[str, list[CatalogEntry]] = {}
    needed_types = set(_APM_CATEGORY_TO_CATALOG_TYPE.values())

    for item in all_items:
        entry = item.to_entry(cfg.asset_base_url)
        if entry is None:
            continue
        inv_type = _infer_type(item)
        if inv_type and inv_type in needed_types:
            by_type.setdefault(inv_type, []).append(entry)

    found = {t for t, v in by_type.items() if v}
    missing = needed_types - found
    if missing:
        logger.warning("Catalog missing entries for types: %s", sorted(missing))

    return CatalogIndex(by_type=by_type)


# ── Type inference ────────────────────────────────────────────────────────────

_KEYWORD_TYPE_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bed", ("bed", "giuong")),
    ("nightstand", ("nightstand", "bedside", "tu_dau_giuong", "tu_canh_giuong")),
    ("wardrobe", ("wardrobe", "closet", "tu_quan_ao", "tu_ao")),
    ("bookshelf", ("bookshelf", "bookcase", "shelf", "ke_sach")),
    ("desk", ("desk", "ban_lam_viec")),
    ("dresser", ("dresser", "dressing", "chest_of_drawers", "tu_ngan_keo")),
    ("dining_table", ("dining_table", "ban_an")),
    ("coffee_table", ("coffee_table", "ban_tra", "ban_cafe")),
    ("tv_console", ("tv_console", "tv_stand", "media_console", "ke_tv", "ke_tivi")),
    ("side_table", ("side_table", "end_table", "ban_phu")),
    ("stool", ("stool", "ghe_don")),
    ("armchair", ("armchair", "lounge_chair", "ghe_sofa_don")),
    ("sofa", ("sofa", "couch", "loveseat", "ghe_sofa")),
    ("chair", ("chair", "ghe_tua", "dining_chair")),
    ("cabinet", ("cabinet", "tu_tru", "wine_cabinet")),
    ("ceiling_light", ("ceiling_lamp", "pendant_lamp", "den_tran", "den_tha")),
)


def _infer_type(item: _CatalogItem) -> str | None:
    if item.object_role:
        low = item.object_role.lower().replace("-", "_").replace(" ", "_")
        for inv_type, keywords in _KEYWORD_TYPE_MAP:
            if any(kw in low for kw in keywords):
                return inv_type

    haystack = " ".join(
        x.lower()
        for x in [item.name or "", item.name_vn or ""]
        if x
    ).replace("-", "_").replace(" ", "_")

    for inv_type, keywords in _KEYWORD_TYPE_MAP:
        if any(kw in haystack for kw in keywords):
            return inv_type

    return None


def _settings_from_env() -> CatalogClientSettings:
    return CatalogClientSettings(
        api_base_url=os.getenv("TKNT_CATALOG_API_BASE_URL", DEFAULT_API_BASE_URL),
        asset_base_url=os.getenv("TKNT_CATALOG_ASSET_BASE_URL", DEFAULT_ASSET_BASE_URL),
        page_limit=int(os.getenv("TKNT_CATALOG_API_PAGE_LIMIT", str(DEFAULT_PAGE_LIMIT))),
        timeout_seconds=float(
            os.getenv("TKNT_CATALOG_API_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
        ),
    )
