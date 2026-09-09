#!/usr/bin/env python3
from __future__ import annotations

"""Print/apply one whitelisted slim-policy stage to an isolated temp runtime.

No service is started and no source runtime, credential or memory is copied.
Defaults to a JSON change preview; --apply is restricted to a temp-directory
runtime with config.json. This helper cannot change a production data directory.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

MAINTENANCE = {
    "daily-knowledge": ("consolidation", "knowledge_mutation_enabled"),
    "weekly": ("consolidation", "weekly_enabled"),
    "monthly": ("consolidation", "monthly_enabled"),
    "self-correction": ("consolidation", "knowledge_self_correction_enabled"),
    "facts": ("rag", "facts_extraction_enabled"),
    "distillation": ("consolidation", "weekly_distillation_enabled"),
    "downscaling": ("consolidation", "synaptic_downscaling_enabled"),
    "skill-learning": ("consolidation", "skill_autolearn_enabled"),
    "curator": ("consolidation", "curator_auto_apply_enabled"),
}


def _object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def validate_sandbox(path: Path) -> Path:
    root = path.resolve()
    temporary = Path(tempfile.gettempdir()).resolve()
    production = Path.home() / ".animaworks"
    configured = os.environ.get("ANIMAWORKS_DATA_DIR", "")
    if root == temporary or not root.is_relative_to(temporary):
        raise ValueError("The sandbox must be a dedicated directory beneath the OS temp directory")
    if root == production.resolve() or (configured and root == Path(configured).resolve()):
        raise ValueError("Refusing the configured production runtime; use a separate offline fixture")
    if not (root / "config.json").is_file():
        raise ValueError("Initialize an offline test runtime with config.json first")
    if not (root / "config.json").resolve().is_relative_to(root):
        raise ValueError("Sandbox config must not link to another runtime")
    for candidate in root.rglob("*.pid"):
        try:
            pid = int(candidate.read_text().strip())
            if pid > 0:
                os.kill(pid, 0)
                raise ValueError("Stop all sandbox processes before applying policy changes")
        except (OSError, TypeError):
            continue
    return root


def changes(root: Path, stage: str, animas: list[str], feature: str | None) -> dict[Path, dict[str, Any]]:
    updates: dict[Path, dict[str, Any]] = {}
    if stage == "priming":
        updates[root / "config.json"] = {
            "prompt": {"system_prompt_target_tokens": 6000},
            "priming": {"profile": "full", "max_tokens": 2000, "heartbeat_context_pct": 0.0},
        }
    elif stage == "maintenance":
        if feature not in MAINTENANCE:
            raise ValueError("Choose one --maintenance-feature to measure independently")
        section, key = MAINTENANCE[feature]
        updates[root / "config.json"] = {section: {key: False}}
    elif stage == "heartbeat":
        acceptance = _object(root / "state" / "slim-runtime-acceptance.json")
        required = ("durable_task_resume", "message_delivery", "cron_delivery", "deadline_delivery", "cancel")
        if any(acceptance.get(key) is not True for key in required):
            raise ValueError(
                "Heartbeat reduction requires passed resume/message/cron/deadline/cancel acceptance checks"
            )
    if stage in {"priming", "heartbeat"}:
        if not animas:
            raise ValueError("Select a small cohort with --anima")
        for name in animas:
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError("Anima names must be plain directory names")
            status = root / "animas" / name / "status.json"
            if not status.resolve().is_relative_to(root) or not status.is_file():
                raise ValueError("The selected anima must already exist inside the sandbox")
            updates[status] = {"priming_profile": "compact"} if stage == "priming" else {"heartbeat_enabled": False}
    return updates


def _merge(data: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(data)
    for key, value in patch.items():
        if isinstance(value, dict):
            existing = merged.get(key, {})
            if not isinstance(existing, dict):
                raise ValueError(f"Expected object for {key}")
            merged[key] = _merge(existing, value)
        else:
            merged[key] = value
    return merged


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    fd, filename = tempfile.mkstemp(prefix=".slim-", suffix=".json", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(filename, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox-runtime", type=Path, required=True)
    parser.add_argument("--stage", choices=("priming", "maintenance", "heartbeat"), required=True)
    parser.add_argument("--anima", action="append", default=[])
    parser.add_argument("--maintenance-feature", choices=tuple(MAINTENANCE))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        root = validate_sandbox(args.sandbox_runtime)
        updates = changes(root, args.stage, args.anima, args.maintenance_feature)
        # Validate every target before any write. Output only whitelisted
        # policy fields, never full configs containing credentials.
        merged = {path: _merge(_object(path), patch) for path, patch in updates.items()}
        if args.apply:
            for path, value in merged.items():
                _atomic_json(path, value)
        print(
            json.dumps(
                {
                    "applied": args.apply,
                    "patches": {str(path.relative_to(root)): patch for path, patch in updates.items()},
                },
                indent=2,
            )
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
