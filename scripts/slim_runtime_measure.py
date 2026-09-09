"""Paired, synthetic framework measurements; no model/provider/network calls."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo


async def probe(source: Path, scratch: Path) -> dict:
    sys.path.insert(0, str(source))
    os.environ["ANIMAWORKS_DATA_DIR"] = str(scratch / ".animaworks")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from tests.helpers.filesystem import create_anima_dir, create_test_data_dir

    data = create_test_data_dir(scratch)
    anima_dir = create_anima_dir(data, "fixture", model="openai/fixture-model", execution_mode="A")
    config_path = data / "config.json"
    config = json.loads(config_path.read_text())
    config["locale"] = "ja"
    config["rag"].update(enabled=False, facts_extraction_enabled=False)
    config_path.write_text(json.dumps(config))
    from core.config import invalidate_cache
    from core.prompt.builder import build_system_prompt
    from core.prompt.tokens import estimate_tokens
    from core.schemas import CycleResult, ModelConfig

    invalidate_cache()
    memory = MagicMock()
    memory.anima_dir = anima_dir
    for name, value in {
        "read_identity": "# Fixture\nI am a synthetic operations worker.",
        "read_injection": "Verify evidence and request approval before external actions.",
        "read_permissions": "",
        "read_specialty_prompt": "",
        "read_bootstrap": "",
        "read_company_vision": "",
        "read_current_state": "status: idle",
        "read_pending": "",
        "read_resolutions": [],
        "list_knowledge_files": [],
        "list_episode_files": [],
        "list_procedure_files": [],
        "list_skill_metas": [],
        "list_common_skill_metas": [],
        "list_procedure_metas": [],
        "list_shared_users": [],
        "read_model_config": ModelConfig(model="openai/fixture-model"),
    }.items():
        getattr(memory, name).return_value = value
    prompts = []
    for mode in ("a", "c"):
        for trigger in ("chat", "inbox:fixture", "heartbeat", "task:fixture"):
            for recall in ("empty", "small_12_pointers", "oversized_plain_recall"):
                content = (
                    ""
                    if recall == "empty"
                    else "\n".join(
                        f'- Known policy {i}: verify approval. read_memory_file(path="knowledge/policy-{i}.md")'
                        for i in range(12 if recall == "small_12_pointers" else 230)
                    )
                )
                with patch(
                    "core.prompt.builder.now_local",
                    return_value=datetime(2026, 9, 8, 12, tzinfo=ZoneInfo("Asia/Tokyo")),
                ):
                    rendered = build_system_prompt(
                        memory,
                        execution_mode=mode,
                        context_window=200_000,
                        trigger=trigger,
                        message="Verify the task",
                        priming_section=content,
                    ).system_prompt
                # Runtime paths are not semantic content and must have identical lengths.
                rendered = rendered.replace(str(data), "/fixture/.animaworks").replace(str(source), "/fixture/source")
                prompts.append(
                    {
                        "mode": mode,
                        "trigger": trigger,
                        "recall_fixture": recall,
                        "characters": len(rendered),
                        "estimated_tokens": estimate_tokens(rendered),
                        "input_recall_estimated_tokens": estimate_tokens(content),
                        "retained_policy_pointers": rendered.count('read_memory_file(path="knowledge/policy-'),
                    }
                )

    from core.memory.priming.engine import PrimingEngine

    channel_names = {
        "A": "_channel_a_sender_profile",
        "B": "_channel_b_recent_activity",
        "C0": "_channel_c0_important_knowledge",
        "C": "_channel_c_related_knowledge",
        "E": "_channel_e_pending_tasks",
        "outbound": "_collect_recent_outbound",
        "F": "_channel_f_episodes",
        "G": "_channel_g_graph_context",
        "notifications": "_collect_pending_human_notifications",
    }
    priming = []
    profiles = (
        ["full", "compact"] if "profile" in inspect.signature(PrimingEngine.prime_memories).parameters else ["full"]
    )
    for profile in profiles:
        for channel, intent, message in (
            ("chat", "question", "What is the approval policy?"),
            ("heartbeat", "", "No monitoring changes"),
            ("inbox", "report", "No monitoring changes"),
        ):
            engine = PrimingEngine(anima_dir)
            counters = {}
            for label, name in channel_names.items():
                text = f"{label}: synthetic memory pointer " * 8
                counter = AsyncMock(return_value=(text, "") if label == "C" else text)
                setattr(engine, name, counter)
                counters[label] = counter
            kwargs = {"profile": profile} if "profile" in inspect.signature(engine.prime_memories).parameters else {}
            result = await engine.prime_memories(message, channel=channel, intent=intent, **kwargs)
            invoked = [label for label, mock in counters.items() if mock.await_count]
            priming.append(
                {
                    "profile": profile,
                    "channel": channel,
                    "intent": intent,
                    "invoked_channels": invoked,
                    "channel_calls": len(invoked),
                    "general_search_channels": [label for label in ("B", "C", "F", "G") if label in invoked],
                    "output_characters": result.total_chars(),
                    "estimated_tokens": result.estimated_tokens(),
                }
            )

    from core._anima_lifecycle import LifecycleMixin
    from core.memory.consolidation import ConsolidationEngine

    engine = ConsolidationEngine(anima_dir, "fixture")
    engine.collect_activity_chunks = MagicMock(return_value=["One unchanged synthetic day of activity."])
    engine.extract_facts_from_text_outcome = AsyncMock(return_value=SimpleNamespace(facts_extracted=0, facts_failed=0))
    engine._find_merge_candidates = MagicMock(return_value=[])
    agent = SimpleNamespace(
        run_cycle=AsyncMock(
            return_value=CycleResult(trigger="consolidation:daily", action="completed", summary="fixture")
        )
    )
    anima = SimpleNamespace(
        name="fixture", anima_dir=anima_dir, memory=memory, agent=agent, _run_autonomous_skill_learning=lambda: None
    )
    consolidation = []
    with patch(
        "core.memory._llm_utils.one_shot_completion",
        new_callable=AsyncMock,
        return_value="## 12:00 — Work\nVerified synthetic evidence",
    ) as generation:
        for number in (1, 2):
            previous_generation, previous_loops = generation.await_count, agent.run_cycle.await_count
            result = await LifecycleMixin._run_daily_consolidation(anima, engine)
            consolidation.append(
                {
                    "run": number,
                    "action": result.action,
                    "episode_generation_calls": generation.await_count - previous_generation,
                    "knowledge_agent_cycles": agent.run_cycle.await_count - previous_loops,
                }
            )
    return {"system_prompts": prompts, "priming": priming, "unchanged_daily_consolidation": consolidation}


def source_counts(source: Path) -> dict:
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "core", "cli", "server"],
        cwd=source,
        text=True,
    ).splitlines()
    files = sorted({name for name in names if name.endswith(".py") and (source / name).is_file()})
    modules = {}
    for root in ("core", "cli", "server"):
        paths = [source / name for name in files if name.startswith(root + "/")]
        modules[root] = {
            "files": len(paths),
            "physical_lines": sum(len(p.read_text().splitlines()) for p in paths),
            "bytes": sum(p.stat().st_size for p in paths),
        }
    return {
        "modules": modules,
        "files": len(files),
        "physical_lines": sum(v["physical_lines"] for v in modules.values()),
        "included_new_task_modules": [name for name in files if name.startswith("core/taskboard/")],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--current", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--baseline-quiet", type=Path)
    parser.add_argument("--baseline-unfiltered", type=Path)
    parser.add_argument("--current-sandbox", type=Path)
    args = parser.parse_args()
    if args.probe:
        with tempfile.TemporaryDirectory(prefix="slim-measure-") as root:
            result = asyncio.run(probe(args.probe, Path(root)))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    result = {
        "method": "same synthetic inputs; production prompt/priming/consolidation control paths; memory searches and all model calls mocked",
        "locale": "ja",
        "fixture_role": "operations worker",
        "fixture_model": "openai/fixture-model",
        "context_window": 200000,
        "token_measure": "repository CJK/other character heuristic, not provider tokens",
        "code_measure": "physical Python lines, tracked plus nonignored untracked core/cli/server files, excludes UI/assets/tests/scripts",
        "model_quality": "not measured",
        "financial_cost": "not measured",
    }
    for label, source in (("baseline", args.baseline), ("current", args.current)):
        with tempfile.TemporaryDirectory(prefix="slim-report-") as root:
            output = Path(root) / "probe.json"
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--probe", str(source), "--output", str(output)],
                cwd=source,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr[-8000:])
            result[label] = {
                "source": str(source),
                "source_counts": source_counts(source),
                **json.loads(output.read_text()),
            }
    result["cron"] = {}
    for label, path in (
        ("baseline_quiet", args.baseline_quiet),
        ("baseline_unfiltered", args.baseline_unfiltered),
        ("current_quiet", args.current_sandbox),
    ):
        if path:
            recorded = json.loads(path.read_text())
            result["cron"][label] = {
                "artifact": str(path),
                "cases": recorded["cron_cases"],
                "matches": sum(bool(row["model_calls"]) == row["expected_followup"] for row in recorded["cron_cases"]),
            }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output)}))


if __name__ == "__main__":
    main()
