from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from pathlib import Path

from core.schemas import ModelConfig

logger = logging.getLogger("animaworks.memory")

# ── ConfigReader ──────────────────────────────────────────


class ConfigReader:
    """Model configuration reader (config.json with config.md fallback)."""

    def __init__(self, anima_dir: Path) -> None:
        self._anima_dir = anima_dir

    def _read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def read_model_config(self) -> ModelConfig:
        """Load model config from unified config.json, with config.md fallback."""
        from core.config import (
            get_config_path,
            load_config,
            resolve_execution_mode,
            resolve_anima_config,
        )

        config_path = get_config_path()
        if config_path.exists():
            config = load_config(config_path)
            anima_name = self._anima_dir.name
            resolved, credential = resolve_anima_config(config, anima_name, anima_dir=self._anima_dir)
            # Derive env var name from credential name (e.g. "anthropic" -> "ANTHROPIC_API_KEY")
            cred_name = resolved.credential
            api_key_env = f"{cred_name.upper()}_API_KEY"
            mode = resolve_execution_mode(
                config, resolved.model, resolved.execution_mode,
            )
            return ModelConfig(
                model=resolved.model,
                fallback_model=resolved.fallback_model,
                max_tokens=resolved.max_tokens,
                max_turns=resolved.max_turns,
                api_key=credential.api_key or None,
                api_key_env=api_key_env,
                api_base_url=credential.base_url,
                context_threshold=resolved.context_threshold,
                max_chains=resolved.max_chains,
                conversation_history_threshold=resolved.conversation_history_threshold,
                execution_mode=resolved.execution_mode,
                supervisor=resolved.supervisor,
                speciality=resolved.speciality,
                resolved_mode=mode,
                thinking=resolved.thinking,
                llm_timeout=resolved.llm_timeout,
            )

        # Legacy fallback: parse config.md
        return self._read_model_config_from_md()

    def _read_model_config_from_md(self) -> ModelConfig:
        """Legacy parser for config.md (fallback when config.json absent)."""
        raw = self._read(self._anima_dir / "config.md")
        if not raw:
            return ModelConfig()

        # Ignore 備考/設定例 sections to avoid matching example lines
        for marker in ("## 備考", "### 設定例"):
            idx = raw.find(marker)
            if idx != -1:
                raw = raw[:idx]

        def _extract(key: str, default: str) -> str:
            m = re.search(rf"^-\s*{key}\s*:\s*(.+)$", raw, re.MULTILINE)
            return m.group(1).strip() if m else default

        defaults = ModelConfig()
        base_url = _extract("api_base_url", "")
        return ModelConfig(
            model=_extract("model", defaults.model),
            fallback_model=_extract("fallback_model", "") or defaults.fallback_model,
            max_tokens=int(_extract("max_tokens", str(defaults.max_tokens))),
            max_turns=int(_extract("max_turns", str(defaults.max_turns))),
            api_key_env=_extract("api_key_env", defaults.api_key_env),
            api_base_url=base_url or defaults.api_base_url,
        )

    def resolve_api_key(self, config: ModelConfig | None = None) -> str | None:
        """Resolve the actual API key (config.json direct value, then env var fallback)."""
        cfg = config or self.read_model_config()
        if cfg.api_key:
            return cfg.api_key
        return os.environ.get(cfg.api_key_env)
