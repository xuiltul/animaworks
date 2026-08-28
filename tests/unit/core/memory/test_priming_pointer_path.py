# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Regression tests for priming pointer path generation (channel C).

Company knowledge indexed in the shared collection must keep its
``companies/`` path — prefixing ``common_knowledge/`` produces a path
read_memory_file cannot resolve (mei daily File-not-found loop, 2026-08).
"""

from core.memory.priming.channel_c import _path_from_doc_id, to_read_memory_path


def test_shared_company_path_not_prefixed() -> None:
    meta = {"anima": "shared", "source_file": "companies/a/knowledge/org-notification-to-owner.md"}
    assert to_read_memory_path(meta, "mei") == "companies/a/knowledge/org-notification-to-owner.md"


def test_shared_plain_source_gets_common_knowledge_prefix() -> None:
    meta = {"anima": "shared", "source_file": "operations/worktree-placement-rule.md"}
    assert to_read_memory_path(meta, "mei") == "common_knowledge/operations/worktree-placement-rule.md"


def test_shared_already_prefixed_source_unchanged() -> None:
    meta = {"anima": "shared", "source_file": "common_knowledge/operations/rule.md"}
    assert to_read_memory_path(meta, "mei") == "common_knowledge/operations/rule.md"


def test_own_anima_source_unchanged() -> None:
    meta = {"anima": "mei", "source_file": "knowledge/beeper-unread-monitoring-policy.md"}
    assert to_read_memory_path(meta, "mei") == "knowledge/beeper-unread-monitoring-policy.md"


def test_doc_id_companies_marker_wins_over_knowledge() -> None:
    # companies/<c>/knowledge/... must not be truncated to knowledge/...
    doc_id = "/data/companies/a/knowledge/org-notification-to-owner.md#0"
    assert _path_from_doc_id(doc_id) == "companies/a/knowledge/org-notification-to-owner.md"
