"""
LLM-based furniture completeness checker using Azure OpenAI.

Given the current furniture list and room context (type, style, description,
special notes, area), asks the LLM which furniture types are still missing
and in what priority order.  Falls back to a simple rule-based list when
Azure credentials are not set or the API call fails.

Environment variables (same as backend/config/openai_config.py):
  AZURE_OPENAI_API_KEY      Azure OpenAI API key
  AZURE_OPENAI_ENDPOINT     e.g. https://<resource>.openai.azure.com
  AZURE_OPENAI_API_VERSION  e.g. 2024-02-15-preview  (default: 2024-02-15-preview)
  AZURE_OPENAI_CHAT_DEPLOYMENT  deployment name (e.g. gpt-4o-mini)

Return value is a list of dicts:
    [{"type": str, "priority": int, "reason": str}, ...]
where ``type`` is a catalog type key, ``priority`` is 1 (highest) … N.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Sequence

logger = logging.getLogger(__name__)

# ── Rule-based fallbacks ──────────────────────────────────────────────────────

_ROOM_DEFAULTS: dict[str, list[str]] = {
    "bedroom": ["bed", "nightstand", "wardrobe", "desk", "chair"],
    "livingroom": ["sofa", "coffee_table", "tv_console", "armchair", "bookshelf"],
    "diningroom": ["dining_table", "chair", "cabinet", "bookshelf"],
}

_FALLBACK_PRIORITIES: dict[str, int] = {
    "bed": 1, "nightstand": 2, "wardrobe": 3, "desk": 4,
    "sofa": 1, "tv_console": 2, "coffee_table": 3, "armchair": 4,
    "dining_table": 1,
    "chair": 3, "bookshelf": 5, "cabinet": 6, "ceiling_light": 7,
}


def _rule_based(
    room_type: str,
    existing_types: set[str],
    available_types: set[str],
    max_items: int,
) -> list[dict]:
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


# ── Azure OpenAI path ─────────────────────────────────────────────────────────

_DEFAULT_API_VERSION = "2024-02-15-preview"

_SYSTEM_PROMPT = """You are an interior design assistant checking whether a furniture layout is complete.

Given a room description and the furniture already placed, identify which furniture types are MISSING.
Only suggest types from the provided available catalog types list.
Do NOT suggest a type that is already in existing_types.

Respond with a JSON array (and nothing else) like:
[
  {"type": "<catalog_type>", "priority": <1=highest>, "reason": "<short reason>"},
  ...
]

Rules:
- Output ONLY the JSON array, no markdown fences, no extra text.
- "priority" must be an integer starting from 1 (most important).
- "reason" must be ≤ 20 words.
- Only use types from available_catalog_types.
- Limit to at most {max_items} items.
"""


def _llm_missing(
    api_key: str,
    endpoint: str,
    api_version: str,
    deployment: str,
    room_type: str,
    style: str | None,
    description: str | None,
    special_notes: str | None,
    existing_types: set[str],
    room_area_m2: float,
    available_types: set[str],
    max_items: int,
) -> list[dict]:
    from openai import AzureOpenAI  # lazy import

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

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

    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=512,
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content or ""
    raw = raw.strip()

    # Azure json_object mode wraps in {"items": [...]} sometimes — handle both
    data = json.loads(raw)
    if isinstance(data, dict):
        # Try common wrapper keys
        for key in ("items", "furniture", "missing", "result", "list"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            # Fallback: flatten dict values if they are dicts
            data = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []

    if not isinstance(data, list):
        raise ValueError(f"Unexpected LLM response shape: {type(data)}")

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
    azure_api_key: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str | None = None,
    azure_deployment: str | None = None,
    max_items: int = 8,
) -> list[dict]:
    """
    Return a prioritised list of missing furniture types.

    Credentials fall back to environment variables if not provided:
      AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
      AZURE_OPENAI_API_VERSION, AZURE_OPENAI_CHAT_DEPLOYMENT
    """
    existing_set = set(existing_catalog_types)
    available_set = set(available_catalog_types) - existing_set

    if not available_set:
        return []

    key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_version = (
        azure_api_version
        or os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
        or _DEFAULT_API_VERSION
    )
    deployment = (
        azure_deployment
        or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
        or os.getenv("AZURE_OPENAI_PRIMARY_DEPLOYMENT", "").strip()
    )

    if not key or not endpoint or not deployment:
        logger.info(
            "Azure OpenAI credentials incomplete (key=%s, endpoint=%s, deployment=%s) "
            "— using rule-based furniture completion.",
            bool(key), bool(endpoint), bool(deployment),
        )
        return _rule_based(room_type, existing_set, available_set, max_items)

    try:
        return _llm_missing(
            api_key=key,
            endpoint=endpoint,
            api_version=api_version,
            deployment=deployment,
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
        logger.warning("Azure OpenAI furniture completion failed (%s) — falling back to rules.", exc)
        return _rule_based(room_type, existing_set, available_set, max_items)
