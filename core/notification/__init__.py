from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Human notification subsystem.

Provides ``HumanNotifier`` and channel implementations for sending
notifications from top-level Animas to human administrators via
external messaging services (Slack, LINE, Telegram, Chatwork, ntfy).
"""

from core.notification.notifier import HumanNotifier, NotificationChannel

__all__ = ["HumanNotifier", "NotificationChannel"]
