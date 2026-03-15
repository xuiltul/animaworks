#!/usr/bin/env python3
"""SWE-bench multi-agent evaluation runner.

Usage:
    # Setup team + run with a test problem
    python3 swe/runner.py --setup-team --test

    # Run SWE-bench Verified instances
    python3 swe/runner.py --run --instances 5

    # List available instances
    python3 swe/runner.py --list
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

# ── Defaults ──────────────────────────────────────────────────


DEFAULT_PORT = 18502
CHAT_TIMEOUT = 600.0  # 10 minutes per chat turn
INSTANCE_TIMEOUT = 1800  # 30 minutes per instance


# ── SSE Client ────────────────────────────────────────────────


async def chat_sse(
    base_url: str,
    anima_name: str,
    message: str,
    timeout: float = CHAT_TIMEOUT,
) -> str:
    """Send chat message and collect response from SSE stream.

    Parses event:/data: pairs properly.
    """
    url = f"{base_url}/api/animas/{anima_name}/chat/stream"
    full_text = ""

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30.0)) as client:
        async with client.stream("POST", url, json={"message": message}) as resp:
            resp.raise_for_status()
            current_event = ""
            async for raw_line in resp.aiter_lines():
                if raw_line.startswith("event: "):
                    current_event = raw_line[7:].strip()
                    continue
                if not raw_line.startswith("data: "):
                    continue
                try:
                    payload = json.loads(raw_line[6:])
                except json.JSONDecodeError:
                    continue

                if current_event == "text_delta":
                    full_text += payload.get("text", "")
                elif current_event == "done":
                    break
                elif current_event == "error":
                    logger.error("SSE error from %s: %s", anima_name, payload)
                    break

    return full_text.strip()


# ── Server Management ─────────────────────────────────────────


def start_server(port: int = DEFAULT_PORT, runtime_dir: Path | None = None) -> subprocess.Popen:
    """Start AnimaWorks server in background with isolated runtime.

    Sets ANIMAWORKS_HOME so the server only sees SWE agents,
    never polluting the production ~/.animaworks/.
    """
    logger.info("Starting AnimaWorks server on port %d ...", port)
    env = {**os.environ}
    if runtime_dir:
        env["ANIMAWORKS_DATA_DIR"] = str(runtime_dir)
        logger.info("Using isolated runtime: %s", runtime_dir)
    proc = subprocess.Popen(
        [sys.executable, "-m", "main", "start", "--foreground", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc


async def wait_for_server(base_url: str, max_wait: int = 120) -> bool:
    """Wait until the server health endpoint responds."""
    logger.info("Waiting for server at %s ...", base_url)
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        for i in range(max_wait):
            try:
                r = await client.get(f"{base_url}/api/system/health")
                if r.status_code == 200:
                    logger.info("Server ready (took %ds)", i)
                    return True
            except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout):
                pass
            await asyncio.sleep(1)
    logger.error("Server did not start within %ds", max_wait)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the server process."""
    logger.info("Stopping server (pid=%d) ...", proc.pid)
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    logger.info("Server stopped")


# ── Repository Management ─────────────────────────────────────


def clone_repo(repo: str, base_commit: str, dest: Path) -> Path:
    """Clone a repository and checkout the specified commit."""
    if dest.exists():
        shutil.rmtree(dest)
    logger.info("Cloning %s @ %s ...", repo, base_commit[:8])
    subprocess.run(
        ["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(dest)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", base_commit],
        cwd=str(dest),
        check=True,
        capture_output=True,
    )
    return dest


def git_diff(repo_dir: Path) -> str:
    """Get the current git diff (unstaged + staged) as a patch string."""
    result = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    return result.stdout


# ── Multi-Agent Orchestration ─────────────────────────────────


async def solve_instance(
    base_url: str,
    instance: dict,
    work_dir: Path,
) -> dict:
    """Solve a single SWE-bench instance using the multi-agent team.

    Flow:
      1. Architect (Opus/Sonnet, Mode S) — analyzes & implements
      2. Investigator (Qwen, Mode A) — second opinion on analysis
      3. Architect — implements fix incorporating feedback
      4. Reviewer (GPT, Mode A) — validates the patch
    """
    instance_id = instance["instance_id"]
    problem = instance["problem_statement"]
    repo_dir = work_dir / instance_id.replace("/", "__")

    logger.info("━━━ Solving: %s ━━━", instance_id)

    # Clone repository
    clone_repo(instance["repo"], instance["base_commit"], repo_dir)

    # Phase 1: Architect analyzes the problem
    logger.info("[Phase 1] Architect analyzing ...")
    analysis = await chat_sse(
        base_url,
        "swe-architect",
        f"""You are solving a SWE-bench problem. The repository has been cloned to:
{repo_dir}

## Problem (GitHub Issue)

{problem}

## Instructions

1. Explore the repository to understand its structure
2. Find the files relevant to this issue
3. Understand the root cause of the bug
4. Write a concise analysis with:
   - Root cause
   - Affected file(s) and line(s)
   - Proposed fix approach

Do NOT implement the fix yet. Just analyze.""",
    )
    logger.info("[Phase 1] Analysis complete (%d chars)", len(analysis))

    # Phase 2: Investigator provides second opinion
    logger.info("[Phase 2] Investigator reviewing ...")
    investigation = await chat_sse(
        base_url,
        "swe-investigator",
        f"""## Problem (GitHub Issue)

{problem}

## Architect's Analysis

{analysis[:8000]}

## Your Task

Review the Architect's analysis above. In 3-5 sentences:
1. Do you agree with the root cause identified?
2. Any blind spots or edge cases the Architect might have missed?
3. Any alternative fix approach worth considering?

Be concise and specific.""",
    )
    logger.info("[Phase 2] Investigation complete (%d chars)", len(investigation))

    # Phase 3: Architect implements the fix
    logger.info("[Phase 3] Architect implementing ...")
    await chat_sse(
        base_url,
        "swe-architect",
        f"""Now implement the fix based on your analysis and the Investigator's feedback.

## Investigator Feedback

{investigation[:4000]}

## Instructions

1. Make the MINIMAL changes needed to fix the issue
2. The repository is at: {repo_dir}
3. Edit the source files directly
4. After making changes, run `cd {repo_dir} && git diff HEAD` to verify your changes
5. If possible, run the relevant tests to validate

Focus on correctness. Do not refactor unrelated code.""",
    )

    # Get the patch
    patch = git_diff(repo_dir)
    logger.info("[Phase 3] Implementation complete (patch: %d chars)", len(patch))

    if not patch.strip():
        logger.warning("Empty patch for %s", instance_id)
        return {
            "instance_id": instance_id,
            "model_name_or_path": "animaworks-team",
            "model_patch": "",
        }

    # Phase 4: Reviewer validates
    logger.info("[Phase 4] Reviewer validating ...")
    review = await chat_sse(
        base_url,
        "swe-reviewer",
        f"""## Problem (GitHub Issue)

{problem}

## Proposed Patch

```diff
{patch[:12000]}
```

## Your Task

Review this patch against the issue description:
1. Does it fix the root cause?
2. Are there any regressions or incomplete fixes?
3. Rate: APPROVE or NEEDS_CHANGES (with specific feedback)

Be concise.""",
    )
    logger.info("[Phase 4] Review: %s", review[:200])

    final_patch = git_diff(repo_dir)

    return {
        "instance_id": instance_id,
        "model_name_or_path": "animaworks-team",
        "model_patch": final_patch,
    }


# ── Test Problem ──────────────────────────────────────────────


def create_test_instance(work_dir: Path) -> dict:
    """Create a simple synthetic test problem to validate the pipeline."""
    repo_dir = work_dir / "test-repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Initialize a git repo with a buggy function
    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo_dir), capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=str(repo_dir), capture_output=True,
    )

    # Create a buggy Python file
    (repo_dir / "calculator.py").write_text('''\
"""Simple calculator module."""


def divide(a, b):
    """Divide a by b.

    Should return a float result and handle division by zero
    by raising a ValueError with a descriptive message.
    """
    return a / b


def average(numbers):
    """Return the average of a list of numbers.

    Should handle empty lists by raising a ValueError.
    """
    total = sum(numbers)
    return total / len(numbers)
''')

    # Create a test file
    (repo_dir / "test_calculator.py").write_text('''\
"""Tests for calculator module."""
import pytest
from calculator import divide, average


def test_divide_basic():
    assert divide(10, 2) == 5.0


def test_divide_by_zero():
    """This test currently FAILS because divide doesn't handle zero."""
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)


def test_average_basic():
    assert average([1, 2, 3]) == 2.0


def test_average_empty():
    """This test currently FAILS because average doesn't handle empty list."""
    with pytest.raises(ValueError, match="Cannot compute average of empty list"):
        average([])
''')

    subprocess.run(["git", "add", "."], cwd=str(repo_dir), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial: buggy calculator"],
        cwd=str(repo_dir), capture_output=True, check=True,
    )

    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_dir), capture_output=True, text=True, check=True,
    ).stdout.strip()

    return {
        "instance_id": "test__calculator-001",
        "repo": "_local_",
        "base_commit": base_commit,
        "problem_statement": (
            "## Bug: divide() and average() don't handle edge cases\n\n"
            "The `divide` function raises `ZeroDivisionError` when dividing by zero. "
            "It should raise `ValueError` with the message 'Cannot divide by zero'.\n\n"
            "The `average` function raises `ZeroDivisionError` on an empty list. "
            "It should raise `ValueError` with the message 'Cannot compute average of empty list'.\n\n"
            "Fix both functions to handle these edge cases properly."
        ),
        "_local_repo_dir": str(repo_dir),
    }


async def solve_test_instance(
    base_url: str,
    work_dir: Path,
) -> dict:
    """Run the pipeline on a synthetic test problem."""
    instance = create_test_instance(work_dir)
    repo_dir = Path(instance["_local_repo_dir"])

    logger.info("━━━ Test Problem: %s ━━━", instance["instance_id"])

    # Phase 1: Architect analyzes
    logger.info("[Phase 1] Architect analyzing ...")
    analysis = await chat_sse(
        base_url,
        "swe-architect",
        f"""You are solving a coding problem. The repository is at:
{repo_dir}

## Problem

{instance['problem_statement']}

## Instructions

1. Read the files in the repository
2. Understand what needs to change
3. Write a brief analysis of the fix needed

Do NOT implement yet.""",
    )
    logger.info("[Phase 1] Done (%d chars)", len(analysis))

    # Phase 2: Investigator second opinion
    logger.info("[Phase 2] Investigator reviewing ...")
    investigation = await chat_sse(
        base_url,
        "swe-investigator",
        f"""## Problem

{instance['problem_statement']}

## Architect's Analysis

{analysis[:6000]}

In 2-3 sentences: agree/disagree? Any blind spots?""",
    )
    logger.info("[Phase 2] Done (%d chars)", len(investigation))

    # Phase 3: Architect implements
    logger.info("[Phase 3] Architect implementing ...")
    await chat_sse(
        base_url,
        "swe-architect",
        f"""Now implement the fix.

Investigator feedback: {investigation[:3000]}

Repository: {repo_dir}
Edit the files directly. Make minimal changes.
After editing, run: cd {repo_dir} && python -m pytest test_calculator.py -v""",
    )

    patch = git_diff(repo_dir)
    logger.info("[Phase 3] Patch (%d chars)", len(patch))

    # Phase 4: Reviewer
    logger.info("[Phase 4] Reviewer ...")
    review = await chat_sse(
        base_url,
        "swe-reviewer",
        f"""## Problem
{instance['problem_statement']}

## Patch
```diff
{patch[:8000]}
```

APPROVE or NEEDS_CHANGES? Be brief.""",
    )
    logger.info("[Phase 4] Review: %s", review[:200])

    # Verify tests pass
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_calculator.py", "-v"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    logger.info("Test result (exit=%d):\n%s", test_result.returncode, test_result.stdout[-500:])

    final_patch = git_diff(repo_dir)
    return {
        "instance_id": instance["instance_id"],
        "model_name_or_path": "animaworks-team",
        "model_patch": final_patch,
        "_test_passed": test_result.returncode == 0,
        "_review": review[:500],
    }


# ── SWE-bench Dataset ────────────────────────────────────────


def load_swe_instances(count: int = 5) -> list[dict]:
    """Load SWE-bench Verified instances from HuggingFace."""
    from datasets import load_dataset

    logger.info("Loading SWE-bench Verified dataset ...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    instances = [dict(ds[i]) for i in range(min(count, len(ds)))]
    logger.info("Loaded %d instances", len(instances))
    return instances


def list_instances(count: int = 20) -> None:
    """Print available SWE-bench instances."""
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    print(f"SWE-bench Verified: {len(ds)} instances\n")
    print(f"{'#':<4} {'instance_id':<50} {'repo':<30}")
    print("─" * 84)
    for i, row in enumerate(ds):
        if i >= count:
            print(f"... ({len(ds) - count} more)")
            break
        print(f"{i:<4} {row['instance_id']:<50} {row['repo']:<30}")


# ── Main ──────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> int:
    base_url = f"http://localhost:{args.port}"

    if args.list:
        list_instances(args.count or 20)
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup team in isolated runtime (never touches ~/.animaworks/)
    runtime_dir = None
    if args.setup_team:
        from swe.team_setup import setup_team

        _, runtime_dir = setup_team()
        logger.info("Isolated runtime: %s", runtime_dir)

    # Start server with isolated ANIMAWORKS_HOME
    server_proc = None
    if not args.no_server:
        server_proc = start_server(args.port, runtime_dir=runtime_dir)
        if not await wait_for_server(base_url):
            if server_proc:
                stop_server(server_proc)
            return 1

    try:
        if args.test:
            # Run synthetic test
            with tempfile.TemporaryDirectory(prefix="swe-test-") as tmp:
                result = await solve_test_instance(base_url, Path(tmp))
                print("\n" + "━" * 60)
                print("TEST RESULT")
                print("━" * 60)
                print(f"  Instance:    {result['instance_id']}")
                print(f"  Patch size:  {len(result['model_patch'])} chars")
                print(f"  Tests pass:  {result.get('_test_passed', 'N/A')}")
                print(f"  Review:      {result.get('_review', 'N/A')[:100]}")
                print("━" * 60)

                # Save result
                out = RESULTS_DIR / "test_result.json"
                out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
                print(f"\nSaved to {out}")

            return 0 if result.get("_test_passed") else 1

        if args.run:
            # Run SWE-bench instances
            instances = load_swe_instances(args.instances)
            predictions = []

            with tempfile.TemporaryDirectory(prefix="swe-run-") as tmp:
                for i, inst in enumerate(instances):
                    logger.info("Instance %d/%d: %s", i + 1, len(instances), inst["instance_id"])
                    try:
                        result = await solve_instance(base_url, inst, Path(tmp))
                        predictions.append(result)
                    except Exception:
                        logger.exception("Failed: %s", inst["instance_id"])
                        predictions.append({
                            "instance_id": inst["instance_id"],
                            "model_name_or_path": "animaworks-team",
                            "model_patch": "",
                        })

            # Save predictions
            out = RESULTS_DIR / "predictions.jsonl"
            with open(out, "w") as f:
                for pred in predictions:
                    f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            print(f"\nPredictions saved to {out} ({len(predictions)} entries)")

            # Summary
            non_empty = sum(1 for p in predictions if p["model_patch"].strip())
            print(f"Patches generated: {non_empty}/{len(predictions)}")

            return 0

    finally:
        if server_proc:
            stop_server(server_proc)

    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="SWE-bench multi-agent evaluation runner")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="AnimaWorks server port")
    p.add_argument("--setup-team", action="store_true", help="Create the agent team before running")
    p.add_argument("--no-server", action="store_true", help="Don't start/stop server (assume already running)")
    p.add_argument("--test", action="store_true", help="Run synthetic test problem")
    p.add_argument("--run", action="store_true", help="Run SWE-bench Verified instances")
    p.add_argument("--list", action="store_true", help="List available instances")
    p.add_argument("--instances", type=int, default=5, help="Number of instances to run")
    p.add_argument("--count", type=int, help="Number of instances to list")
    args = p.parse_args()

    if not any([args.test, args.run, args.list, args.setup_team]):
        p.print_help()
        sys.exit(1)

    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
