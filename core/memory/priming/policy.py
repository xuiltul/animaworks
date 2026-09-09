from __future__ import annotations

"""Small, model-independent policy resolved before priming does any searches."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrimingPolicy:
    profile: str = "compact"
    dynamic_budget: bool = True
    max_tokens: int = 2000


def resolve_priming_policy(anima_dir: Path) -> PrimingPolicy:
    from core.config import load_config

    try:
        config = load_config().priming
        profile = config.profile
        dynamic = config.dynamic_budget
        maximum = config.max_tokens
    except Exception:
        logger.debug("Using default priming policy", exc_info=True)
        profile, dynamic, maximum = "compact", True, 2000
    try:
        status = json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))
        override = status.get("priming_profile") if isinstance(status, dict) else None
        if override in ("compact", "full"):
            profile = override
    except (OSError, ValueError, TypeError):
        pass
    return PrimingPolicy(
        profile=profile if profile in ("compact", "full") else "compact",
        dynamic_budget=dynamic if isinstance(dynamic, bool) else True,
        max_tokens=maximum if isinstance(maximum, int) and maximum >= 200 else 2000,
    )
