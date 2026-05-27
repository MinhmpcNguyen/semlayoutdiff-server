"""
Catalog API client — fetches furniture items from the external catalog.

Provides a pre-loaded index keyed by semantic category name so the pipeline
adapter can resolve category → (catalogItemId, modelUrl, size, color) without
a per-request network call.
"""
from __future__ import annotations

import logging
import os
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "https://auto-furniture-api2.a-star.group"
DEFAULT_ASSET_BASE_URL = "https://storage.mazig.io"
DEFAULT_PAGE_LIMIT = 500
DEFAULT_TIMEOUT_SECONDS = 20.0

# Maps APM category names → catalog type key.
# Fallback rules (for types not in catalog):
#   armchair / lounge_chair → chair  (no standalone armchair in catalog)
#   coffee_table / side_table / corner_side_table / round_end_table → table (bàn gỗ)
#   dressing_table / dresser → desk  (no bàn phấn in catalog; desk is closest)
#   stool → chair                    (no stool in catalog)
_APM_CATEGORY_TO_CATALOG_TYPE: dict[str, str] = {
    # Beds
    "kids_bed": "bed",
    "single_bed": "bed",
    "double_bed": "bed",
    # Tables — generic wooden table (bàn gỗ) used as fallback
    "coffee_table": "table",
    "corner_side_table": "table",
    "round_end_table": "table",
    "side_table": "table",
    # TV / console
    "console_table": "tv_console",
    "tv_stand": "tv_console",
    # Desk / work table
    "desk": "desk",
    "dressing_table": "desk",   # no bàn phấn in catalog → use desk
    # Dining
    "table": "dining_table",
    "dining_table": "dining_table",
    # Seating — no armchair/stool in catalog → fall back to chair
    "stool": "chair",
    "dressing_chair": "chair",
    "dining_chair": "chair",
    "chinese_chair": "chair",
    "armchair": "chair",
    "lounge_chair": "chair",
    "chair": "chair",
    # Sofas
    "loveseat_sofa": "sofa",
    "lazy_sofa": "sofa",
    "sofa": "sofa",
    "multi_seat_sofa": "sofa",
    "chaise_longue_sofa": "sofa",
    "l_shaped_sofa": "sofa",
    # Bedroom storage
    "nightstand": "nightstand",
    "wardrobe": "wardrobe",
    # Shelving / storage
    "shelf": "bookshelf",
    "bookshelf": "bookshelf",
    "children_cabinet": "cabinet",
    "wine_cabinet": "cabinet",
    "cabinet": "cabinet",
    # Lighting
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

    def get_by_type(
        self,
        catalog_type: str,
        max_w_m: float | None = None,
        max_d_m: float | None = None,
    ) -> list[CatalogEntry]:
        """
        Return catalog entries of the given type, optionally filtered by size.

        Parameters
        ----------
        catalog_type : key from ``_KEYWORD_TYPE_MAP`` (e.g. "sofa", "bed")
        max_w_m : maximum width in metres (dimension index 0 of size_m)
        max_d_m : maximum depth in metres (dimension index 2 of size_m)
        """
        entries = self._by_type.get(catalog_type, [])
        if max_w_m is None and max_d_m is None:
            return list(entries)
        result = []
        for e in entries:
            if e.size_m is None:
                continue
            w, _h, d = e.size_m
            if max_w_m is not None and w > max_w_m:
                continue
            if max_d_m is not None and d > max_d_m:
                continue
            result.append(e)
        return result

    def available_types(self) -> list[str]:
        """Return all catalog type keys that have at least one entry."""
        return [t for t, v in self._by_type.items() if v]

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

# Keywords are in ASCII-normalized form (no Vietnamese diacritics).
# Listed from most-specific to least-specific so the first match wins.
_KEYWORD_TYPE_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Nightstands first — name contains "giuong" but must NOT be classified as bed
    ("nightstand",   ("tu_dau_giuong", "tu_canh_giuong", "nightstand", "bedside")),
    # Beds
    ("bed",          ("giuong_forest", "giuong_rustic", "giuong_ngu", "giuong",
                      "bed", "double_bed", "single_bed")),
    # Wardrobe
    ("wardrobe",     ("tu_quan_ao", "wardrobe", "closet")),
    # Desk (before generic table keywords)
    ("desk",         ("ban_lam_viec", "ban_go_lam_viec", "desk")),
    # TV console
    ("tv_console",   ("ke_tv", "ke_tivi", "tv_stand", "media_console", "tv_console")),
    # Dining table
    ("dining_table", ("ban_an", "dining_table")),
    # Generic table — "bàn gỗ" (ban_go) used for coffee/side/dressing fallback
    # Excludes ban_bep (kitchen counter, only 3cm tall)
    ("table",        ("ban_go_forest", "ban_go")),
    # Coffee table (dedicated, in case catalog adds one later)
    ("coffee_table", ("ban_tra", "coffee_table", "ban_cafe")),
    # Side table
    ("side_table",   ("ban_phu", "side_table", "end_table")),
    # Dresser
    ("dresser",      ("ban_phan", "tu_ngan_keo", "dresser", "dressing")),
    # Bookshelf – catalog has "kệ gỗ" (ke_go)
    ("bookshelf",    ("ke_go", "ke_sach", "bookshelf", "bookcase", "shelf")),
    # Cabinet – catalog has "tủ gỗ" (tu_go) and "tủ góc" (tu_goc)
    ("cabinet",      ("tu_go", "tu_goc", "cabinet", "tu_tru")),
    # Sofa (before generic chair)
    ("sofa",         ("sofa", "couch", "loveseat", "ghe_sofa")),
    # Armchair
    ("armchair",     ("armchair", "lounge_chair", "ghe_sofa_don")),
    # Stool
    ("stool",        ("stool", "ghe_don")),
    # Chair – "ghe_tua" for named chairs; "ghe" as last-resort catch-all
    ("chair",        ("ghe_tua", "ghe_go", "ghe_forest", "dining_chair", "chair", "ghe")),
    # Ceiling light – catalog has "đèn trần" (den_tran) and "đèn treo" (den_treo)
    ("ceiling_light", ("den_tran", "den_treo", "ceiling_lamp", "pendant_lamp", "den_tha")),
)


def _normalize_vn(text: str) -> str:
    """Strip Vietnamese diacritics and return lowercase ASCII."""
    # 'đ' (U+0111) has no NFKD decomposition → replace explicitly first
    text = text.replace("đ", "d").replace("Đ", "d")
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return normalized.lower().replace("-", "_").replace(" ", "_")


def _infer_type(item: _CatalogItem) -> str | None:
    if item.object_role:
        low = _normalize_vn(item.object_role)
        for inv_type, keywords in _KEYWORD_TYPE_MAP:
            if any(kw in low for kw in keywords):
                return inv_type

    raw = " ".join(x for x in [item.name or "", item.name_vn or ""] if x)
    haystack = _normalize_vn(raw)

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
