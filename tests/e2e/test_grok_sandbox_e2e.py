from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Live E2E coverage for the Mode X Grok filesystem sandbox."""

import json
import shlex
import shutil
from pathlib import Path

import pytest

from core.execution.grok_cli import GrokCLIExecutor
from core.platform.grok import is_grok_authenticated, is_grok_cli_available
from core.schemas import ModelConfig

pytestmark = [pytest.mark.e2e, pytest.mark.live, pytest.mark.timeout(300)]


@pytest.fixture
def use_mock(request: pytest.FixtureRequest) -> bool:
    """Override the shared Anthropic-key live gate with Grok authentication."""
    return request.config.getoption("--mock") or not is_grok_authenticated()


async def test_grok_profile_enforces_deny_and_write_roots(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """The real Grok CLI can write only allowed roots and cannot read denies."""
    if not is_grok_cli_available():
        pytest.skip("Grok CLI is not installed")
    if not is_grok_authenticated():
        pytest.skip("Grok CLI is not authenticated; run grok login")

    # Do not place the outside-write canary under /tmp: Grok's workspace
    # profile intentionally grants writes to temporary directories.
    runtime_root = Path.cwd() / f".e2e-grok-sandbox-{tmp_path.name}"
    runtime_root.mkdir()
    request.addfinalizer(lambda: shutil.rmtree(runtime_root, ignore_errors=True))

    anima_dir = runtime_root / "animas" / "grok-sandbox"
    denied_dir = runtime_root / "denied"
    anima_dir.mkdir(parents=True)
    denied_dir.mkdir()
    (anima_dir / "identity.md").write_text("# Grok sandbox E2E\n", encoding="utf-8")

    secret = denied_dir / "secret.txt"
    secret.write_text("GROK_DENIED_CANARY\n", encoding="utf-8")
    (anima_dir / "permissions.json").write_text(
        json.dumps(
            {
                "version": 1,
                "file_roots": [str(anima_dir)],
                "file_roots_denied": [str(denied_dir)],
            }
        ),
        encoding="utf-8",
    )

    denied_stdout = anima_dir / "denied.stdout"
    denied_stderr = anima_dir / "denied.stderr"
    denied_status = anima_dir / "denied.status"
    outside_file = runtime_root / "outside-write.txt"
    outside_stderr = anima_dir / "outside.stderr"
    outside_status = anima_dir / "outside.status"
    inside_file = anima_dir / "inside-write.txt"

    def quote(path: Path) -> str:
        return shlex.quote(str(path))

    denied_command = f"cat {quote(secret)}"
    outside_command = f"printf 'OUTSIDE_WRITE\\n' > {quote(outside_file)}"
    script = "\n".join(
        [
            f"/bin/sh -c {shlex.quote(denied_command)} > {quote(denied_stdout)} 2> {quote(denied_stderr)}",
            f"printf '%s\\n' \"$?\" > {quote(denied_status)}",
            f"/bin/sh -c {shlex.quote(outside_command)} 2> {quote(outside_stderr)}",
            f"printf '%s\\n' \"$?\" > {quote(outside_status)}",
            f"printf 'INSIDE_WRITE\\n' > {quote(inside_file)}",
        ]
    )
    prompt = (
        "Use your shell or terminal tool to run exactly this POSIX shell script once. "
        "Do not simulate the commands and do not change any paths.\n\n"
        f"{script}"
    )
    model_config = ModelConfig(
        model="grok/grok-4.5",
        max_tokens=4096,
        max_turns=30,
        credential="grok",
        context_threshold=0.5,
        max_chains=2,
    )
    executor = GrokCLIExecutor(model_config, anima_dir)

    result = await executor.execute(
        prompt,
        system_prompt="Follow the user's shell-test instructions exactly.",
        trigger="heartbeat",
    )
    lowered_result = result.text.lower()
    if any(word in lowered_result for word in ("unauthenticated", "not authenticated", "run grok login")):
        pytest.skip("Grok CLI credentials are unavailable or expired")

    assert inside_file.read_text(encoding="utf-8") == "INSIDE_WRITE\n", result.text
    assert int(denied_status.read_text(encoding="utf-8")) != 0
    denied_error = denied_stderr.read_text(encoding="utf-8").lower()
    assert "permission denied" in denied_error or "operation not permitted" in denied_error
    assert "GROK_DENIED_CANARY" not in denied_stdout.read_text(encoding="utf-8")
    assert int(outside_status.read_text(encoding="utf-8")) != 0
    assert not outside_file.exists()
    assert result.result_message is not None
    assert result.result_message.session_id
