"""Run bounded real server/worker smoke behind Docker network isolation.

The local HTTP model is deterministic. This measures execution plumbing and
cron call suppression, NOT real-model quality, financial ROI or acceptance.
Only the source tree, its Python environment, and a scratch output directory
are mounted. No runtime, credentials, Docker socket, or host network access.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock


async def exercise_watcher(app: Any, anima_dir: Path, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Publish through HTTP; inspect durable facts without bypassing execution."""
    import httpx

    from core.memory.task_queue import TaskQueueManager

    store = TaskQueueManager(anima_dir).store

    def snapshot() -> dict[str, Any]:
        with store.reader() as db:
            tasks = {
                r["task_id"]: {
                    "status": json.loads(r["entry_json"])["status"],
                    "ready": bool(r["ready"]),
                    "active": r["current_attempt"] is not None,
                }
                for r in db.execute("SELECT * FROM tasks WHERE anima='sandbox'")
            }
            attempts = [
                dict(r)
                for r in db.execute(
                    "SELECT task_id,number,started_at,ended_at,stop_kind FROM task_attempts WHERE anima='sandbox' ORDER BY started_at"
                )
            ]
            wakeups = [
                dict(r)
                for r in db.execute("SELECT task_id,reason,acknowledged_at FROM task_wakeups WHERE anima='sandbox'")
            ]
        return {"tasks": tasks, "attempts": attempts, "wakeups": wakeups}

    async def until(predicate: Any, label: str) -> dict[str, Any]:
        for _ in range(300):
            state = snapshot()
            if predicate(state):
                return state
            await asyncio.sleep(0.2)
        raise AssertionError(f"Watcher timeout: {label}: {snapshot()}")

    def ended(state: dict[str, Any], task_id: str, count: int = 1) -> bool:
        rows = [r for r in state["attempts"] if r["task_id"] == task_id]
        return len(rows) == count and all(r["ended_at"] for r in rows)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:19851", trust_env=False, timeout=30) as client:

        async def submit(tasks: list[dict[str, Any]]) -> None:
            tasks = [{"task_type": "llm", **task} if not task.get("resume") else task for task in tasks]
            response = await client.post("/api/internal/submit-tasks", json={"anima_name": "sandbox", "tasks": tasks})
            assert response.status_code == 200, response.text

        initial = [
            {
                "task_id": "waiting",
                "title": "Undeclared fixture",
                "description": "SANDBOX_PENDING:waiting. Do not use any tools; respond only with fixture output.",
            },
            {
                "task_id": "dependent",
                "title": "Dependency fixture",
                "description": "SANDBOX_DECLARE:dependent",
                "depends_on": ["waiting"],
                "reply_to": "sandbox",
            },
        ]
        store.pause_claims("sandbox")
        try:
            await submit(initial)
            await submit(initial)
        finally:
            store.pause_claims("sandbox", paused=False)
        first = await until(
            lambda s: (
                ended(s, "waiting") and any(w["task_id"] == "waiting" and w["acknowledged_at"] for w in s["wakeups"])
            ),
            "undeclared pending and wake delivery",
        )
        assert first["tasks"]["waiting"] == {"status": "pending", "ready": False, "active": False}
        assert not any(r["task_id"] == "dependent" for r in first["attempts"])
        with store.reader() as db:
            old_token = db.execute("SELECT token FROM task_attempts WHERE task_id='waiting' AND number=1").fetchone()[0]
        # Cross two watcher polls. An LLM return is not completion or implicit retry.
        await asyncio.sleep(7)
        assert ended(snapshot(), "waiting")
        pid_before = app.state.supervisor.processes["sandbox"].get_pid()
        await app.state.supervisor.restart_anima("sandbox")
        pid_after = app.state.supervisor.processes["sandbox"].get_pid()
        assert pid_before != pid_after
        await asyncio.sleep(3)
        assert ended(snapshot(), "waiting")
        await submit([{"task_id": "waiting", "resume": True}])
        assert not store.finish(old_token, status="done", stop_kind="stale_test")
        await until(lambda s: ended(s, "waiting", 2), "explicit resume exactly once")
        response = await client.post(
            "/api/internal/update-task", json={"anima_name": "sandbox", "task_id": "waiting", "status": "done"}
        )
        assert response.status_code == 200, response.text
        completed = await until(
            lambda s: ended(s, "dependent") and s["tasks"]["dependent"]["status"] == "done",
            "dependency tool declaration",
        )
        parent_end = [r for r in completed["attempts"] if r["task_id"] == "waiting"][-1]["ended_at"]
        child_start = next(r["started_at"] for r in completed["attempts"] if r["task_id"] == "dependent")
        assert child_start >= parent_end
        store.pause_claims("sandbox")
        try:
            await submit(
                [{"task_id": "cancelled", "title": "Cancelled before start", "description": "Must never execute"}]
            )
            response = await client.post(
                "/api/internal/update-task",
                json={"anima_name": "sandbox", "task_id": "cancelled", "status": "cancelled"},
            )
            assert response.status_code == 200, response.text
        finally:
            store.pause_claims("sandbox", paused=False)
        await submit(
            [
                {
                    "task_id": "running-cancel",
                    "title": "Quiet provider cancellation",
                    "description": "SANDBOX_QUIET:running-cancel",
                }
            ]
        )
        await until(
            lambda s: (
                s["tasks"]["running-cancel"]["active"] and any(c["task_fixture"] == "running-cancel" for c in calls)
            ),
            "quiet provider has request",
        )
        cancellation_started = asyncio.get_running_loop().time()
        response = await client.post(
            "/api/internal/update-task",
            json={"anima_name": "sandbox", "task_id": "running-cancel", "status": "cancelled"},
        )
        assert response.status_code == 200, response.text
        await until(lambda s: ended(s, "running-cancel"), "running cancellation without model chunks")
        cancellation_seconds = asyncio.get_running_loop().time() - cancellation_started
        assert cancellation_seconds < 12, cancellation_seconds
        await app.state.supervisor.restart_anima("sandbox")
        await asyncio.sleep(3)
        final = snapshot()
        assert ended(final, "waiting", 2) and ended(final, "dependent")
        assert final["tasks"]["cancelled"]["status"] == "cancelled"
        assert not any(r["task_id"] == "cancelled" for r in final["attempts"])
        assert any(c["declared_task"] == "dependent" for c in calls)
        task_calls = {
            task_id: sum(c["task_fixture"] == task_id for c in calls)
            for task_id in ("waiting", "dependent", "running-cancel")
        }
        assert task_calls == {"waiting": 2, "dependent": 2, "running-cancel": 1}, task_calls
        return {
            **final,
            "http_submission": True,
            "real_watcher": True,
            "resume_attempts": 2,
            "duplicate_ready_submission_one_attempt": True,
            "stale_finish_rejected": True,
            "quiet_provider_cancel_seconds": cancellation_seconds,
            "heartbeat_enabled": False,
            "model_calls_by_task": task_calls,
            "dependency_order_verified": True,
            "restart_no_replay": True,
            "worker_restart_pids": [pid_before, pid_after, app.state.supervisor.processes["sandbox"].get_pid()],
            "done_source": {
                "waiting": "explicit HTTP operator update after ended attempt",
                "dependent": "Mode A update_task tool",
            },
        }


async def run_inside(output: Path, cases_path: Path, monitor_profile: str) -> dict[str, Any]:
    import httpx
    import uvicorn

    from core.config import invalidate_cache
    from tests.helpers.filesystem import create_anima_dir, create_test_data_dir

    root = Path(tempfile.mkdtemp(prefix="slim-runtime-", dir=output.parent))
    data = create_test_data_dir(root)
    os.environ["ANIMAWORKS_DATA_DIR"] = str(data)
    os.environ["ANIMAWORKS_EMBED_URL"] = "http://127.0.0.1:19851/api/internal/embed"
    calls: list[dict[str, Any]] = []
    canonical = (Path.cwd() / "core/taskboard/tasks.py").exists()

    class ModelEndpoint(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = body.get("messages", [])
            last_user = next((str(m.get("content", "")) for m in reversed(messages) if m["role"] == "user"), "")
            marker = re.search(r"SANDBOX_DECLARE:([a-z-]+)", last_user)
            task_marker = re.search(r"SANDBOX_(?:DECLARE|PENDING|QUIET):([a-z-]+)", last_user)
            tool_name = next(
                (
                    t["function"]["name"]
                    for t in body.get("tools", [])
                    if t.get("function", {}).get("name") == "update_task"
                ),
                None,
            )
            declaration = bool(marker and tool_name and messages[-1]["role"] != "tool")
            message: dict[str, Any] = {"role": "assistant", "content": "Sandbox result recorded. No external action."}
            if declaration:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-sandbox",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(
                                    {"task_id": marker[1], "status": "done", "result": "Verified deterministic fixture"}
                                ),
                            },
                        }
                    ],
                }
            calls.append(
                {
                    "messages": len(body.get("messages", [])),
                    "prompt_characters": sum(len(str(m.get("content", ""))) for m in body.get("messages", [])),
                    "declared_task": marker[1] if declaration else None,
                    "task_fixture": task_marker[1] if task_marker else None,
                }
            )
            if "SANDBOX_QUIET:running-cancel" in last_user:
                threading.Event().wait(20)
            response = {
                "id": "sandbox-response",
                "object": "chat.completion",
                "created": 1,
                "model": "sandbox-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if declaration else "stop",
                        "message": message,
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            }
            if body.get("stream"):
                delta = dict(message)
                if "tool_calls" in delta:
                    delta["tool_calls"][0]["index"] = 0
                chunk = {
                    **response,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                terminal = {
                    **chunk,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": response["choices"][0]["finish_reason"]}],
                }
                encoded = f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(terminal)}\n\ndata: [DONE]\n\n".encode()
            else:
                encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if body.get("stream") else "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when the quiet-provider request is cancelled.

    endpoint = ThreadingHTTPServer(("127.0.0.1", 19852), ModelEndpoint)
    threading.Thread(target=endpoint.serve_forever, daemon=True).start()
    anima_dir = create_anima_dir(
        data,
        "sandbox",
        model="openai/sandbox-model",
        execution_mode="A" if canonical else "B",
        credential="sandbox",
        api_key="sandbox-not-a-secret",
        api_base_url="http://127.0.0.1:19852/v1",
    )
    config_path = data / "config.json"
    config = json.loads(config_path.read_text())
    config.update(
        {
            "setup_complete": True,
            "rag": {
                "enabled": False,
                "vector_worker_enabled": False,
                "startup_repair_preflight_enabled": False,
                "rerank_enabled": False,
            },
            "external_messaging": {"slack": {"enabled": False}, "chatwork": {"enabled": False}},
            "external_tasks": {"enabled": False},
        }
    )
    config["anima_defaults"].update(
        {
            "heartbeat_enabled": False,
            "consolidation_enabled": False,
            "model": "openai/sandbox-model",
            "credential": "sandbox",
        }
    )
    config["credentials"]["sandbox"] = {"api_key": "sandbox-not-a-secret", "base_url": "http://127.0.0.1:19852/v1"}
    config_path.write_text(json.dumps(config))
    (data / "permissions.global.json").write_text(
        (Path.cwd() / "templates/_shared/config_defaults/permissions.global.json").read_text()
    )
    status_path = anima_dir / "status.json"
    status = json.loads(status_path.read_text())
    status.update({"heartbeat_enabled": False, "consolidation_enabled": False, "process_model": "legacy"})
    status_path.write_text(json.dumps(status))
    # These are anonymous sandbox settings, never a copy of a live agent's cron.
    (anima_dir / "cron.md").write_text(
        "## temperature\nschedule: */10 * * * *\ntype: command\ncommand: printf OK\n"
        "skip_pattern: \\AOK\\Z\ntrigger_heartbeat: true\n"
    )
    invalidate_cache()
    import server.app as server_module

    # The runtime and worker lifecycle are real; optional native-model warmup
    # and avatar/catalog network jobs are excluded from this hermetic smoke.
    for name in ("_run_model_warmup", "_warm_model_catalog", "_warm_voice_greets", "_reconcile_assets_at_startup"):
        setattr(server_module, name, AsyncMock())

    app = server_module.create_app(data / "animas", data / "shared")
    app.state.listen_port = 19851
    app.state.supervisor.child_env_urls = {"ANIMAWORKS_EMBED_URL": os.environ["ANIMAWORKS_EMBED_URL"]}
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=19851, log_level="warning"))
    server_task = asyncio.create_task(server.serve())
    results: dict[str, Any] = {
        "data_dir": str(data),
        "quality_evaluation": "not_run_deterministic_model",
        "monitor_profile": monitor_profile,
        "server_mode": "runtime lifespan; optional native-model/avatar/catalog warmups mocked",
    }
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            for _ in range(100):
                try:
                    health = await client.get("http://127.0.0.1:19851/health")
                    if health.status_code == 200:
                        break
                    results["last_health_status"] = health.status_code
                    results["last_health_body"] = health.text[:200]
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.1)
            else:
                raise RuntimeError("HTTP server did not become healthy")
        results["http_health"] = health.status_code
        await asyncio.wait_for(app.state._anima_startup_task, 90)
        from core import startup_progress

        assert startup_progress.snapshot()["phase"] == "ready", startup_progress.snapshot()
        handle = app.state.supervisor.processes["sandbox"]
        results["worker_pid"] = handle.get_pid()
        before = len(calls)
        prompt_chars_before = sum(c["prompt_characters"] for c in calls)
        reply = await app.state.supervisor.send_request(
            anima_name="sandbox",
            method="run_cron_task",
            params={"task_name": "sandbox-smoke", "task_description": "Record sandbox result. Do not use tools."},
            timeout=60,
        )
        results["worker_task_reply"] = reply
        results["worker_task_model_calls"] = len(calls) - before
        results["worker_task_prompt_characters"] = sum(c["prompt_characters"] for c in calls) - prompt_chars_before
        if len(calls) == before:
            raise AssertionError("worker task never reached the local model endpoint")
        await app.state.supervisor.restart_anima("sandbox")
        results["restarted_worker_pid"] = app.state.supervisor.processes["sandbox"].get_pid()
        assert results["restarted_worker_pid"] != results["worker_pid"]
        if canonical:
            results["watcher_pipeline"] = await exercise_watcher(app, anima_dir, calls)

        # Real command subprocesses and the production cron contract, with a
        # local HTTP model proxy. This excludes AgentCore prompt/recall costs.
        from core.schemas import CronTask, CycleResult
        from core.supervisor.task_runner import execute_cron_contract

        class CronProbe:
            command_runs = 0

            async def run_cron_command(self, name: str, **kwargs: Any) -> dict[str, Any]:
                self.command_runs += 1
                case = json.loads(kwargs["args"]["case"])
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    "import json,sys; c=json.loads(sys.argv[1]); print(c['stdout'],end=''); "
                    "print(c.get('stderr',''),end='',file=sys.stderr); sys.exit(c['exit_code'])",
                    json.dumps(case),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()
                return {
                    "task": name,
                    "exit_code": process.returncode,
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                }

            async def run_cron_task(self, *args: Any, **kwargs: Any) -> CycleResult:
                async with httpx.AsyncClient(trust_env=False) as client:
                    response = await client.post(
                        "http://127.0.0.1:19852/v1/chat/completions",
                        json={
                            "messages": [{"role": "user", "content": kwargs.get("command_output", "")}],
                        },
                    )
                    response.raise_for_status()
                return CycleResult(trigger="cron", action="responded", summary="deterministic proxy")

        probe = CronProbe()
        rows = []
        cases = json.loads(cases_path.read_text())
        for case in cases:
            if case["kind"] != "cron":
                continue
            task = CronTask(
                name=case["id"],
                type="command",
                schedule="*/10 * * * *",
                tool="fixture",
                args={"case": json.dumps(case)},
                skip_pattern=r"\AOK\Z" if monitor_profile == "quiet" else None,
                trigger_heartbeat=case.get("trigger_heartbeat", True),
            )
            before = len(calls)
            outcome = await execute_cron_contract(probe, task)
            rows.append(
                {
                    "id": case["id"],
                    "model_calls": len(calls) - before,
                    "expected_followup": case["expected_followup"],
                    "success": outcome["success"],
                }
            )
        results.update(
            {
                "cron_cases": rows,
                "command_runs": probe.command_runs,
                "proxy_model_calls": len(calls),
                "proxy_prompt_characters": sum(c["prompt_characters"] for c in calls),
                "quality_cases_available_not_scored": sum(c["kind"] == "quality" for c in cases),
                "cron_contract_matches": sum(bool(row["model_calls"]) == row["expected_followup"] for row in rows),
            }
        )
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, 15)
        endpoint.shutdown()
        output.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--image", default="python:latest")
    parser.add_argument("--monitor-profile", choices=("quiet", "unfiltered"), default="quiet")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.inside:
        sys.path.insert(0, str(args.source))
        print(
            json.dumps(
                asyncio.run(
                    run_inside(args.output, repo / "tests/fixtures/slim_runtime/cases.json", args.monitor_profile)
                ),
                indent=2,
            )
        )
        return
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    python_root = Path(sys._base_executable).resolve().parents[1]
    mounts = {repo, args.source.resolve(), python_root}
    container_name = f"animaworks-slim-{uuid.uuid4().hex[:12]}"
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        container_name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
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
        "LITELLM_LOCAL_MODEL_COST_MAP=True",
        "--env",
        f"PYTHONPATH={args.source.resolve()}",
    ]
    for mount in sorted(mounts):
        command.extend(["--mount", f"type=bind,src={mount},dst={mount},readonly"])
    command.extend(
        [
            "--mount",
            f"type=bind,src={args.output},dst=/results",
            "--entrypoint",
            str(repo / ".venv/bin/python"),
            args.image,
            str(repo / "scripts/slim_runtime_sandbox.py"),
            "--inside",
            "--source",
            str(args.source.resolve()),
            "--output",
            "/results/result.json",
            "--monitor-profile",
            args.monitor_profile,
        ]
    )
    try:
        result = subprocess.run(command, timeout=240, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "stop", "--time", "5", container_name], timeout=20, check=False, capture_output=True)
        raise
    (args.output / "stdout.log").write_text(result.stdout)
    (args.output / "stderr.log").write_text(result.stderr)
    print(json.dumps({"exit_code": result.returncode, "output": str(args.output)}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
