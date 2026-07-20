# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Chatwork external tasks collector (stub)."""

from __future__ import annotations

from core.external_tasks.models import ExternalTask


def collect_chatwork() -> list[ExternalTask]:
    """Collect actionable Chatwork items (e.g. tasks / mentions).

    Task ids MUST use the form ``chatwork-{stable_external_id}``.
    """
    raise NotImplementedError("chatwork external tasks collector is not implemented yet")
