#!/usr/bin/env python3
from __future__ import annotations

"""Paired, bounded real-model reasoning probes through AgentCore mode C.

Only anonymous fixture questions and generated runtime prompts are sent. This
does not measure production outcomes or prove retrieval/implementation quality.
Docker mounts ONLY source, interpreter, Codex binary and one read-only auth
file. Generated runtimes/native sessions live in disposable tmpfs, never reports.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


def score_result(text: str, case: dict[str, Any]) -> dict[str, Any]:
    try:
        # Runtime appends an emotion HTML comment after the answer object.
        answer, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
    except (ValueError, json.JSONDecodeError):
        return {"passed": False, "reason": "invalid_json"}
    if not isinstance(answer, dict):
        return {"passed": False, "reason": "invalid_json_object"}
    decision = answer.get("decision")
    evidence = str(answer.get("evidence", ""))
    passed = decision == case["expected_decision"]
    if case.get("required_evidence"):
        passed = passed and case["required_evidence"] in evidence
    return {"passed": passed, "decision": decision, "evidence": evidence}


async def run_inside(args: argparse.Namespace) -> dict[str, Any]:
    from tests.helpers.filesystem import create_anima_dir, create_test_data_dir

    root = Path(tempfile.mkdtemp(prefix="slim-live-"))
    data = create_test_data_dir(root)
    os.environ["ANIMAWORKS_DATA_DIR"] = str(data)
    os.environ["ANIMAWORKS_EMBED_URL"] = "http://127.0.0.1:1"
    cases = json.loads(args.cases.read_text())[: args.limit]
    config_path = data / "config.json"
    config = json.loads(config_path.read_text())
    config.update(
        {
            "setup_complete": True,
            "rag": {"enabled": False, "facts_extraction_enabled": False, "vector_worker_enabled": False},
            "human_notification": {"enabled": False},
            "external_messaging": {"slack": {"enabled": False}, "chatwork": {"enabled": False}},
            "external_tasks": {"enabled": False},
        }
    )
    if args.profile == "compact":
        config.update(
            {"prompt": {"system_prompt_target_tokens": 6000}, "priming": {"profile": "compact", "max_tokens": 2000}}
        )
    config_path.write_text(json.dumps(config))
    from core.agent import AgentCore
    from core.config import invalidate_cache
    from core.memory import MemoryManager
    from core.schemas import ModelConfig

    report: dict[str, Any] = {
        "source": str(args.source),
        "profile": args.profile,
        "model": args.model,
        "effort": "low",
        "runtime": "AgentCore.run_cycle / native Codex mode C",
        "scope": "Synthetic text reasoning with supplied evidence; no production task/quality claim",
        "cases": [],
    }
    for index, case in enumerate(cases):
        name = f"probe{index}"
        anima_dir = create_anima_dir(
            data,
            name,
            model=args.model,
            execution_mode="C",
            credential="openai",
            identity="# Anonymous evaluator\nYou evaluate synthetic evidence accurately. Never invent facts.",
            injection=(
                "This is a bounded offline-evidence reasoning test. Answer in English. "
                "Use only supplied evidence. Do not run tools, access network, inspect credentials, "
                "send messages, create tasks or modify files. Propose actions without executing them."
            ),
        )
        status = json.loads((anima_dir / "status.json").read_text())
        status.update({"heartbeat_enabled": False, "consolidation_enabled": False, "thinking_effort": "low"})
        if args.profile == "compact":
            status["priming_profile"] = "compact"
        (anima_dir / "status.json").write_text(json.dumps(status))
        invalidate_cache()
        model_config = ModelConfig(
            model=args.model,
            execution_mode="C",
            resolved_mode="C",
            api_key="",
            api_key_env="",
            credential="openai",
            thinking_effort="low",
            max_chains=1,
            max_tokens=2048,
            fallback_models=[],
        )
        agent = AgentCore(anima_dir, MemoryManager(anima_dir), model_config=model_config)
        choices = sorted({entry["expected_decision"] for entry in json.loads(args.cases.read_text())})
        prompt = (
            case["question"] + "\n\nThis is a synthetic reasoning-only question. Do not use tools or take actions. "
            "Respond with a JSON object only, containing decision, evidence and explanation. "
            f"Choose decision from {json.dumps(choices)}. Evidence must cite relevant supplied facts/IDs, "
            "including an ISO8601 timestamp when a deadline is requested. Explanation must be under 100 words."
        )
        started = time.monotonic()
        result_record: dict[str, Any] = {"id": case["id"], "expected_decision": case["expected_decision"]}
        try:
            result = await asyncio.wait_for(
                agent.run_cycle(prompt, trigger="manual", thread_id=case["id"]), args.case_timeout
            )
            result_record.update(
                {
                    "action": result.action,
                    "answer": result.summary,
                    "usage": result.usage,
                    "tool_calls": len(result.tool_call_records or []),
                    "score": score_result(result.summary, case),
                }
            )
            if result.summary.startswith("[Codex SDK Error:"):
                result_record["error_type"] = "CodexExecutionError"
        except Exception as exc:
            result_record.update({"error_type": type(exc).__name__, "score": {"passed": False}})
        result_record["latency_seconds"] = round(time.monotonic() - started, 3)
        instructions = anima_dir / ".codex_home" / "instructions.md"
        result_record["system_prompt_characters"] = len(instructions.read_text()) if instructions.exists() else None
        report["cases"].append(result_record)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    "id": case["id"],
                    "passed": result_record["score"]["passed"],
                    "latency_seconds": result_record["latency_seconds"],
                    "error_type": result_record.get("error_type"),
                }
            ),
            flush=True,
        )
        if result_record.get("error_type") or result_record.get("action") == "error":
            # Authentication/transport failure must not fan out into 24 retries.
            break
    report["passed"] = sum(bool(item["score"]["passed"]) for item in report["cases"])
    report["attempted"] = len(report["cases"])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/slim_runtime/live_quality.json",
    )
    parser.add_argument("--profile", choices=("full", "compact"), default="compact")
    parser.add_argument("--model", default="codex/gpt-6-astra")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--case-timeout", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--codex-vendor",
        type=Path,
        default=Path(
            "/home/main/.local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl"
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 12:
        parser.error("--limit must be between 1 and 12")
    if args.inside:
        sys.path.insert(0, str(args.source))
        logging.basicConfig(level=logging.ERROR)
        asyncio.run(run_inside(args))
        return
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    auth = Path.home() / ".codex/auth.json"
    if not auth.is_file():
        parser.error("Codex auth.json is unavailable; no fallback credential is authorized")
    name = f"slim-live-{uuid.uuid4().hex[:12]}"
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        name,
        "--network",
        "bridge",
        "--read-only",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--memory",
        "4g",
        "--tmpfs",
        "/tmp:rw,nosuid,size=1g",
        "--workdir",
        str(args.source.resolve()),
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        f"PYTHONPATH={args.source.resolve()}",
        "--env",
        "ANIMAWORKS_CODEX_PATH=/opt/codex/bin/codex",
        "--env",
        "ANIMAWORKS_CODEX_BG_IDLE_TIMEOUT_SEC=90",
        "--env",
        "ANIMAWORKS_CODEX_FG_IDLE_TIMEOUT_SEC=90",
    ]
    for mount in {repo, args.source.resolve(), Path(sys._base_executable).resolve().parents[1]}:
        command.extend(["--mount", f"type=bind,src={mount},dst={mount},readonly"])
    # Docker's numeric UID (absent from image passwd) has HOME=/; do not
    # override host HOME or mount any other contents of the user's home.
    for source, target, readonly in (
        (auth, "/.codex/auth.json", True),
        (args.codex_vendor, "/opt/codex", True),
        (output, "/results", False),
    ):
        command.extend(["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")])
    command.extend(
        [
            "--entrypoint",
            str(repo / ".venv/bin/python"),
            "python:latest",
            str(Path(__file__).resolve()),
            "--inside",
            "--source",
            str(args.source.resolve()),
            "--profile",
            args.profile,
            "--model",
            args.model,
            "--limit",
            str(args.limit),
            "--case-timeout",
            str(args.case_timeout),
            "--output",
            "/results/result.json",
        ]
    )
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=args.limit * (args.case_timeout + 30) + 90
        )
        # Provider error text may be useful, but never include credential files/configs.
        (output / "stdout.log").write_text(result.stdout)
        (output / "stderr.log").write_text(result.stderr)
        print(json.dumps({"returncode": result.returncode, "output": str(output)}))
    finally:
        subprocess.run(["docker", "stop", "--time", "3", name], capture_output=True, timeout=15, check=False)


if __name__ == "__main__":
    main()
