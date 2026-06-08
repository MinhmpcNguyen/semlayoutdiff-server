"""
LLM-based furniture completeness checker using Azure OpenAI.

API key resolution (mirrors backend/config/openai_config.py):
  1. AZURE_OPENAI_API_KEY env var (plaintext)
  2. OPENAI_PUBLIC_KEY  (Fernet key)  +  OPENAI_PRIVATE_KEY_FILE  (path to
     the encrypted .enc file) → Fernet.decrypt(file) → plaintext key
  3. OPENAI_API_KEY env var (plaintext fallback)

Other env vars:
  AZURE_OPENAI_ENDPOINT          e.g. https://<resource>.services.ai.azure.com/...
                                  (path portion is stripped automatically)
  AZURE_OPENAI_API_VERSION       default: 2024-12-01-preview
  AZURE_OPENAI_CHAT_DEPLOYMENT   deployment name  (e.g. gpt-5.4-mini-furniture)
  AZURE_OPENAI_PRIMARY_DEPLOYMENT  alias for the above
  OPENAI_API_VERSION             fallback for api version

Return value:
    [{"type": str, "priority": int, "reason": str}, ...]
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Directory of this file — used to resolve relative private_key paths
_SERVER_DIR = Path(__file__).parent.resolve()
_BACKEND_NEW_ROOT = _SERVER_DIR.parent

_DEFAULT_API_VERSION = "2024-12-01-preview"

# ── Fernet key decryption (same logic as openai_config.py) ───────────────────

def _resolve_api_key() -> str | None:
    """
    Resolve the Azure OpenAI API key using the same three-step logic as
    the original backend's OpenAIAccess.get_openai_access().
    """
    def _env(*names: str) -> str | None:
        for n in names:
            v = os.getenv(n, "").strip()
            if v and not (v.startswith("${") and v.endswith("}")):
                return v
        return None

    # 1. Direct plaintext env vars
    key = _env("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")
    if key:
        return key

    # 2. Fernet-encrypted file
    public_key = _env("OPENAI_PUBLIC_KEY")
    key_file = _env("OPENAI_PRIVATE_KEY_FILE")
    if public_key and key_file:
        try:
            from cryptography.fernet import Fernet
            # Resolve path: if relative, look next to this file then backend_new root
            p = Path(key_file)
            if not p.is_absolute():
                candidates = [
                    _BACKEND_NEW_ROOT / p,
                    _SERVER_DIR / p,
                    Path.cwd() / p,
                ]
                for c in candidates:
                    if c.exists():
                        p = c
                        break
            encrypted = p.read_bytes()
            return Fernet(public_key.encode()).decrypt(encrypted).decode()
        except Exception as exc:
            logger.warning("Fernet key decryption failed: %s", exc)

    return None


def _normalize_endpoint(endpoint: str) -> str:
    """Strip path/query from Azure endpoint URL (same as openai_client._normalize_azure_endpoint)."""
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid Azure endpoint: {endpoint!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


# ── Rule-based fallbacks ──────────────────────────────────────────────────────

_ROOM_DEFAULTS: dict[str, list[str]] = {
    "bedroom": ["bed", "nightstand", "wardrobe", "desk", "chair"],
    "livingroom": ["sofa", "coffee_table", "tv_console", "armchair", "bookshelf"],
    "diningroom": ["dining_table", "chair", "cabinet", "bookshelf"],
    # kitchen: LLM fills fridge / sink / cabinets; dining table comes from SLDN
    "kitchen": ["fridge", "sink", "kitchen_cabinet", "microwave", "stove"],
}

_FALLBACK_PRIORITIES: dict[str, int] = {
    "bed": 1, "nightstand": 2, "wardrobe": 3, "desk": 4,
    "sofa": 1, "tv_console": 2, "coffee_table": 3, "armchair": 4,
    "dining_table": 1,
    "chair": 3, "bookshelf": 5, "cabinet": 6, "ceiling_light": 7,
    # kitchen priorities
    "fridge": 1, "sink": 2, "kitchen_cabinet": 3, "stove": 4, "microwave": 5,
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


# ── Azure OpenAI call ─────────────────────────────────────────────────────────

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

    raw = (resp.choices[0].message.content or "").strip()

    data = json.loads(raw)
    # Azure json_object mode sometimes wraps in {"items": [...]}
    if isinstance(data, dict):
        for key in ("items", "furniture", "missing", "result", "list"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
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
    # Optional overrides — fall back to env vars when None
    azure_api_key: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str | None = None,
    azure_deployment: str | None = None,
    max_items: int = 8,
) -> list[dict]:
    """
    Return a prioritised list of missing furniture types.

    API key is resolved (in order):
      1. ``azure_api_key`` argument
      2. AZURE_OPENAI_API_KEY / OPENAI_API_KEY env vars
      3. Fernet decrypt: OPENAI_PUBLIC_KEY + OPENAI_PRIVATE_KEY_FILE

    Endpoint / deployment fall back to env vars:
      AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_CHAT_DEPLOYMENT (or PRIMARY),
      AZURE_OPENAI_API_VERSION / OPENAI_API_VERSION
    """
    existing_set = set(existing_catalog_types)
    available_set = set(available_catalog_types) - existing_set
    if not available_set:
        return []

    def _env(*names: str) -> str | None:
        for n in names:
            v = os.getenv(n, "").strip()
            if v and not (v.startswith("${") and v.endswith("}")):
                return v
        return None

    key = azure_api_key or _resolve_api_key()
    endpoint_raw = azure_endpoint or _env("AZURE_OPENAI_ENDPOINT") or ""
    api_version = (
        azure_api_version
        or _env("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION")
        or _DEFAULT_API_VERSION
    )
    deployment = (
        azure_deployment
        or _env("AZURE_OPENAI_CHAT_DEPLOYMENT", "AZURE_OPENAI_PRIMARY_DEPLOYMENT")
        or ""
    )

    if not key or not endpoint_raw or not deployment:
        logger.info(
            "Azure OpenAI config incomplete (key=%s endpoint=%s deployment=%s) "
            "— using rule-based completion.",
            bool(key), bool(endpoint_raw), bool(deployment),
        )
        return _rule_based(room_type, existing_set, available_set, max_items)

    try:
        endpoint = _normalize_endpoint(endpoint_raw)
    except ValueError as exc:
        logger.warning("Bad Azure endpoint %r: %s — rule-based fallback.", endpoint_raw, exc)
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
        logger.warning("Azure OpenAI call failed (%s) — rule-based fallback.", exc)
        return _rule_based(room_type, existing_set, available_set, max_items)
