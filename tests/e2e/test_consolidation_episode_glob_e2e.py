from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""E2E tests for episode file discovery in consolidation.

Previously tested daily_consolidate() with glob-based suffixed episode files
(YYYY-MM-DD_xxx.md). Those tests were removed because daily_consolidate()
was deleted in the consolidation refactor — consolidation is now Anima-driven
via run_consolidation() with tool-call loops.

The underlying _collect_recent_episodes() method is still present and tested
in the unit test suite (test_consolidation_refactored.py).
"""
