"""
LLM-based furniture completeness checker using Claude claude-haiku-4-5.

Given the current furniture list and room context (type, style, description,
special notes, area), asks Claude which furniture types are still missing and
in what priority order.  Falls back to a simple rule-based list when
``ANTHROPIC_API_KEY`` is not set or the API call fails.

Return value is a list of dicts:
    [{"type": str, "priority": int, "reason": str}, ...]
where ``type`` is a catalog type key (one of the values in
``_APM_CATEGORY_TO_CATALOG_TYPE``), ``priority`` is 1 (highest) … N, and
``reason`` is a short human-readable explanation.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Sequence

logger = logging.getLogger(__name__)

# ── Rule-based fallbacks (used when no API key or LLM fails) ─────────────────

_ROOM_DEFAULTS: dict[str, list[str]] = {
    "bedroom": [
        "bed", "nightstand", "wardrobe", "desk", "chair",
    ],
    "livingroom": [
        "sofa", "coffee_table", "tv_console", "armchair", "bookshelf",
    ],
    "diningroom": [
        "dining_table", "chair", "cabinet", "bookshelf",
    ],
}

_FALLBACK_PRIORITIES: dict[str, int] = {
    # Bedroom
    "bed": 1, "nightstand": 2, "wardrobe": 3, "desk": 4,
    # Living
    "sofa": 1, "tv_console": 2, "coffee_table": 3, "armchair": 4,
    # Dining
    "dining_table": 1,
    # Common
    "chair": 3, "bookshelf": 5, "cabinet": 6, "ceiling_light": 7,
}


def _rule_based(
    room_type: str,
    existing_types: set[str],
    available_types: set[str],
    max_items: int,
) -> list[dict]:
    """Return prioritised list of missing catalog types using static rules."""
    defaults = _ROOM_DEFAULTS.get(room_type, _ROOM_DEFAULTS["livingroom"])
    missing = []
    for t in defaults:
        if t not in existing_types and t in available_types:
            missing.append({
                "type": t,
                "priority": _FALLBACK_PRIORITIES.get(t, 9),
                "reason": f"Standard {room_type} item.",
            })
    missing.sort(key=lambda x: x["priority"])
    return missing[:max_items]


# ── LLM path ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an interior design assistant that checks whether a furniture layout is complete.

Given a room description and the furniture already placed, you must identify which furniture types are MISSING.
Only suggest types from the provided available catalog types list.
Do not suggest a type that is already present in the existing_types list.

Respond with a JSON array (and nothing else) of objects like:
[
  {"type": "<catalog_type>", "priority": <1=highest>, "reason": "<short reason>"},
  ...
]

Rules:
- Output ONLY the JSON array, no markdown fences, no prose.
- "priority" must be an integer starting from 1 (most important).
- "reason" must be ≤ 20 words.
- Do NOT repeat types already in existing_types.
- Only use types from the provided available_catalog_types.
- Limit output to at most {max_items} items.
"""


def _llm_missing(
    api_key: str,
    room_type: str,
    style: str | None,
    description: str | None,
    special_notes: str | None,
    existing_types: set[str],
    room_area_m2: float,
    available_types: set[str],
    max_items: int,
) -> list[dict]:
    import anthropic  # lazy import so server starts even without the package

    client = anthropic.Anthropic(api_key=api_key)

    user_msg = (
        f"Room type: {room_type}\n"
        f"Room area: {room_area_m2:.1f} m²\n"
        f"Style: {style or 'modern'}\n"
        f"Description: {description or '(none)'}\n"
        f"Special notes: {special_notes or '(none)'}\n"
        f"Existing furniture types: {sorted(existing_types) or '(none)'}\n"
        f"Available catalog types: {sorted(available_types)}\n"
        f"Max items to suggest: {max_items}\n"
        "\nList the missing furniture types in priority order."
    )

    system = _SYSTEM_PROMPT.replace("{max_items}", str(max_items))

    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = resp.content[0].text.strip()
    # Strip optional markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"LLM returned non-list: {type(data)}")

    result: list[dict] = []
    for item in data:
        t = str(item.get("type", "")).strip()
        p = int(item.get("priority", 9))
        r = str(item.get("reason", ""))[:120]
        if t and t in available_types and t not in existing_types:
            result.append({"type": t, "priority": p, "reason": r})

    result.sort(key=lambda x: x["priority"])
    return result[:max_items]


# ── Public API ────────────────────────────────────────────────────────────────

def get_missing_furniture(
    room_type: str,
    existing_catalog_types: Sequence[str],
    available_catalog_types: Sequence[str],
    room_area_m2: float = 20.0,
    style: str | None = None,
    description: str | None = None,
    special_notes: str | None = None,
    api_key: str | None = None,
    max_items: int = 8,
) -> list[dict]:
    """
    Return a prioritised list of missing furniture types.

    Parameters
    ----------
    room_type : "bedroom" | "livingroom" | "diningroom"
    existing_catalog_types : catalog type strings already placed in the room
    available_catalog_types : catalog type strings that exist in the catalog
    room_area_m2 : room floor area in m²
    style / description / special_notes : from PipelineNormalizeRunRequest
    api_key : Anthropic API key (falls back to ``ANTHROPIC_API_KEY`` env var)
    max_items : maximum number of suggestions

    Returns
    -------
    list of {"type": str, "priority": int, "reason": str}
    """
    existing_set = set(existing_catalog_types)
    available_set = set(available_catalog_types) - existing_set

    if not available_set:
        return []

    key = api_key or os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        logger.info("No ANTHROPIC_API_KEY — using rule-based furniture completion.")
        return _rule_based(room_type, existing_set, available_set, max_items)

    try:
        return _llm_missing(
            api_key=key,
            room_type=room_type,
            style=style,
            description=description,
            special_notes=special_notes,
            existing_types=existing_set,
            room_area_m2=room_area_m2,
            available_types=available_set,
            max_items=max_items,
        )
    except Exception as exc:
        logger.warning("LLM furniture completion failed (%s) — falling back to rules.", exc)
        return _rule_based(room_type, existing_set, available_set, max_items)
