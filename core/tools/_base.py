# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Base infrastructure for AnimaWorks tools."""
from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("animaworks.tools")


from core.exceptions import ToolConfigError  # noqa: F401 – re-export


@dataclass
class ToolResult:
    """Standardized return value from tool execution."""
    success: bool
    data: Any = None
    text: str = ""
    error: str | None = None


def get_env_or_fail(key: str, tool_name: str) -> str:
    """Get an environment variable, raising a clear error if missing."""
    val = os.environ.get(key)
    if not val:
        raise ToolConfigError(
            f"Tool '{tool_name}' requires environment variable {key}. "
            f"Set it in .env or the shell environment."
        )
    return val


# ── Unified Credential Resolution ────────────────────────────


def get_credential(
    credential_name: str,
    tool_name: str,
    key_name: str = "api_key",
    env_var: str | None = None,
) -> str:
    """Resolve a credential via config.json → shared/credentials.json → env cascade.

    Args:
        credential_name: Key in config.json ``credentials`` dict
            (e.g. ``"chatwork"``).
        tool_name: Human-readable tool name for error messages.
        key_name: Which key to retrieve.  ``"api_key"`` reads the primary
            ``api_key`` field; anything else reads from ``keys[key_name]``.
        env_var: Fallback environment variable name.

    Returns:
        The resolved credential string.

    Raises:
        ToolConfigError: If neither config.json, shared/credentials.json,
            nor the environment variable provides a value.
    """
    from core.config.models import load_config

    # 1. config.json
    config = load_config()
    cred = config.credentials.get(credential_name)
    if cred:
        if key_name == "api_key" and cred.api_key:
            _log_resolved(credential_name, key_name, "config.json", cred.api_key)
            return cred.api_key
        if key_name != "api_key" and key_name in cred.keys and cred.keys[key_name]:
            val = cred.keys[key_name]
            _log_resolved(credential_name, key_name, "config.json", val)
            return val

    # 2. shared/credentials.json
    if env_var:
        val = _lookup_shared_credentials(env_var)
        if val:
            _log_resolved(credential_name, key_name, "shared/credentials.json", val)
            return val

    # 3. Environment variable fallback
    if env_var:
        val = os.environ.get(env_var)
        if val:
            _log_resolved(credential_name, key_name, f"env:{env_var}", val)
            return val

    # 4. Error with guidance
    sources = [f"config.json credentials.{credential_name}.{key_name}"]
    if env_var:
        sources.append("shared/credentials.json")
        sources.append(f"environment variable {env_var}")
    raise ToolConfigError(
        f"Tool '{tool_name}' requires credential '{credential_name}'. "
        f"Set it in: {' or '.join(sources)}"
    )


def _lookup_shared_credentials(key: str) -> str | None:
    """Look up a key in the shared credentials file.

    Reads ``{data_dir}/shared/credentials.json`` (a flat key-value JSON)
    and returns the value for *key*, or ``None`` if not found.
    """
    from core.paths import get_data_dir

    cred_file = get_data_dir() / "shared" / "credentials.json"
    if not cred_file.is_file():
        return None
    try:
        data = json.loads(cred_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", cred_file, exc)
        return None
    val = data.get(key)
    return val if val else None


def _log_resolved(
    credential_name: str, key_name: str, source: str, value: str,
) -> None:
    """Log credential resolution with masked value."""
    masked = value[:4] + "****" if len(value) > 4 else "****"
    logger.debug(
        "Credential '%s.%s' resolved from %s: %s",
        credential_name, key_name, source, masked,
    )


# ── CLI Guide Auto-Generation ────────────────────────────────


def auto_cli_guide(tool_name: str, schemas: list[dict[str, Any]]) -> str:
    """Auto-generate a CLI usage guide from tool schemas.

    Produces a markdown snippet showing ``animaworks-tool`` CLI usage for
    each schema, deriving argument flags from JSON Schema properties.

    Args:
        tool_name: The TOOL_MODULES key (e.g. ``"web_search"``).
        schemas: The list returned by ``get_tool_schemas()``.

    Returns:
        Markdown string with CLI examples.
    """
    lines = [f"### {tool_name}", "```bash"]
    for schema in schemas:
        params = schema.get("input_schema", schema.get("parameters", {}))
        props = params.get("properties", {})
        required = set(params.get("required", []))

        parts = [f"animaworks-tool {tool_name}"]
        # Positional: first required string parameter
        for pname in required:
            prop = props.get(pname, {})
            if prop.get("type") == "string":
                parts.append(f'"<{pname}>"')
                break
        # Optional flags
        for pname, prop in props.items():
            if pname in required:
                continue
            flag = f"--{pname.replace('_', '-')}"
            ptype = prop.get("type", "string")
            if ptype == "boolean":
                parts.append(f"[{flag}]")
            else:
                parts.append(f"[{flag} <{ptype}>]")
        parts.append("-j")
        lines.append(" ".join(parts))
    lines.append("```")
    return "\n".join(lines)


# ── Execution Profile Loading ─────────────────────────────


def load_execution_profiles(
    tool_modules: dict[str, str],
    personal_tools: dict[str, str] | None = None,
) -> dict[str, dict[str, dict[str, object]]]:
    """Load EXECUTION_PROFILE from all tool modules.

    Returns:
        Nested dict: {tool_name: {subcommand: {expected_seconds, background_eligible}}}.
        Tools without EXECUTION_PROFILE are omitted.
    """
    import importlib
    import importlib.util

    profiles: dict[str, dict[str, dict[str, object]]] = {}

    for tool_name, module_path in tool_modules.items():
        try:
            mod = importlib.import_module(module_path)
            if hasattr(mod, "EXECUTION_PROFILE"):
                profiles[tool_name] = mod.EXECUTION_PROFILE
        except Exception:
            logger.debug("Failed to load EXECUTION_PROFILE for %s", tool_name, exc_info=True)

    if personal_tools:
        for tool_name, file_path in personal_tools.items():
            try:
                spec = importlib.util.spec_from_file_location(
                    f"animaworks_profile_{tool_name}", file_path,
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    if hasattr(mod, "EXECUTION_PROFILE"):
                        profiles[tool_name] = mod.EXECUTION_PROFILE
            except Exception:
                logger.debug("Failed to load EXECUTION_PROFILE for personal tool %s", tool_name, exc_info=True)

    return profiles


def get_eligible_tools_from_profiles(
    profiles: dict[str, dict[str, dict[str, object]]],
) -> dict[str, int]:
    """Extract eligible tools map from loaded profiles.

    Returns:
        Dict of {tool_subcommand_key: expected_seconds} for background_eligible=True entries.
        The key format matches BackgroundTaskManager expectations.
    """
    eligible: dict[str, int] = {}
    for tool_name, subcommands in profiles.items():
        for subcmd, info in subcommands.items():
            if info.get("background_eligible"):
                eligible[f"{tool_name}:{subcmd}"] = int(info.get("expected_seconds", 60))
    return eligible
