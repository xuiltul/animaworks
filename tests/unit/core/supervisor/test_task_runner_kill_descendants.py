"""Killing a task-runner job must reach descendants that left its process group.

Codex's sandbox runs shell tools under ``setsid``-style sessions; before
2026-08-30 those survived ``killpg`` as orphans and kept the disk saturated.
"""

from __future__ import annotations

import os
import subprocess
import time
from types import SimpleNamespace

import psutil
import pytest

from core.supervisor.task_runner_supervisor import TaskRunnerJob, TaskRunnerSupervisor


def _job_for(proc: subprocess.Popen) -> TaskRunnerJob:
    job = object.__new__(TaskRunnerJob)
    job.process = SimpleNamespace(returncode=None, send_signal=proc.send_signal)
    job.pid = proc.pid
    job.pgid = os.getpgid(proc.pid)
    return job


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_kill_job_group_reaches_new_session_descendants() -> None:
    # Grandchild in its own session (like codex-linux-sandbox), outside the job pgid.
    proc = subprocess.Popen(
        ["bash", "-c", "setsid bash -c 'echo $$; exec sleep 300' & wait"],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    grandchild_pid = int(proc.stdout.readline().strip())
    assert os.getpgid(grandchild_pid) != os.getpgid(proc.pid)

    TaskRunnerSupervisor._kill_job_group(_job_for(proc))
    proc.wait(timeout=5)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and psutil.pid_exists(grandchild_pid):
        if psutil.Process(grandchild_pid).status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(0.05)
    assert not psutil.pid_exists(grandchild_pid) or psutil.Process(grandchild_pid).status() == psutil.STATUS_ZOMBIE
