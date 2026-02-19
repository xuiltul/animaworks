# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Process isolation supervisor package.

Provides process-level isolation for each Anima by running them in
separate subprocesses communicating via Unix Domain Sockets.
"""

from __future__ import annotations

from core.supervisor.ipc import IPCClient, IPCServer, IPCRequest, IPCResponse, IPCEvent
from core.supervisor.process_handle import ProcessHandle, ProcessState, ProcessStats
from core.supervisor.manager import ProcessSupervisor, RestartPolicy, HealthConfig

__all__ = [
    "IPCClient",
    "IPCServer",
    "IPCRequest",
    "IPCResponse",
    "IPCEvent",
    "ProcessHandle",
    "ProcessState",
    "ProcessStats",
    "ProcessSupervisor",
    "RestartPolicy",
    "HealthConfig",
]
