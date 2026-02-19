"""INTL LoRA Router — O(1) profile → adapter mapping (§3.4).

No inference performed here. Pure lookup that returns the adapter name
and HuggingFace path for a given PROFILE string. Raises RouterError for
unknown profiles.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

_CONFIGS_PATH = Path(__file__).parent.parent / "configs" / "adapters.json"

# ── Static adapter map (profile → adapter metadata) ──────────────────────────
# Loaded from configs/adapters.json at import time; can also be overridden
# via RouterConfig.
ADAPTER_MAP: dict[str, dict] = {}

def _load_adapter_map() -> dict[str, dict]:
    global ADAPTER_MAP
    try:
        raw = json.loads(_CONFIGS_PATH.read_text())
        repo = raw.get("adapter_repo", "")
        base = raw.get("base_model", "Qwen/Qwen2.5-Coder-3B-Instruct")
        adapters = raw.get("adapters", {})
        result: dict[str, dict] = {}
        for profile, meta in adapters.items():
            result[profile] = {
                "profile": profile,
                "phase": meta.get("phase", -1),
                "threshold": meta.get("threshold", 0.90),
                "pairs": meta.get("pairs", 0),
                "adapter_id": f"{repo}/tree/main/{profile}" if repo else profile,
                "base_model": base,
            }
        return result
    except FileNotFoundError:
        return {}

ADAPTER_MAP = _load_adapter_map()


# ── Errors ────────────────────────────────────────────────────────────────────
class RouterError(Exception):
    """Raised when a profile cannot be routed to an adapter."""


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RouteResult:
    profile: str
    phase: int
    threshold: float
    pairs: int
    adapter_id: str    # HuggingFace path
    base_model: str    # Base model HF id


# ── Public API ────────────────────────────────────────────────────────────────
def route(profile: str) -> RouteResult:
    """Return adapter routing info for *profile*.

    Args:
        profile: INTL PROFILE string (e.g. ``"python_fastapi"``).

    Returns:
        :class:`RouteResult` with adapter metadata.

    Raises:
        :class:`RouterError`: If *profile* is not in ADAPTER_MAP.
    """
    if not profile:
        raise RouterError("profile must not be empty")
    entry = ADAPTER_MAP.get(profile)
    if entry is None:
        known = sorted(ADAPTER_MAP.keys())
        raise RouterError(
            f"Unknown INTL profile {profile!r}. "
            f"Known profiles ({len(known)}): {', '.join(known)}"
        )
    return RouteResult(
        profile=entry["profile"],
        phase=entry["phase"],
        threshold=entry["threshold"],
        pairs=entry["pairs"],
        adapter_id=entry["adapter_id"],
        base_model=entry["base_model"],
    )


def list_profiles(phase: int | None = None) -> list[str]:
    """Return all known profiles, optionally filtered by *phase*."""
    if phase is None:
        return sorted(ADAPTER_MAP.keys())
    return sorted(p for p, m in ADAPTER_MAP.items() if m["phase"] == phase)


def is_trained(profile: str) -> bool:
    """Return True if the adapter has been trained (phase exists and >0 pairs)."""
    entry = ADAPTER_MAP.get(profile)
    return entry is not None and entry.get("pairs", 0) > 0
