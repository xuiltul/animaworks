from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

# NOTE: With process isolation, DigitalAnima instances are no longer
# in the parent process. This file provides stub dependencies for
# backwards compatibility during transition.

from typing import Any


def get_anima(anima_name: str) -> Any:
    """
    Stub dependency for compatibility.

    With process isolation, this should not be used.
    Routes should use ProcessSupervisor IPC instead.
    """
    raise NotImplementedError(
        "get_anima() is deprecated with process isolation. "
        "Use ProcessSupervisor IPC instead."
    )
