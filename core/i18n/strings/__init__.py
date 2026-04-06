# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Merge all domain string modules into a single dict."""

from __future__ import annotations


def _merge_strings() -> dict[str, dict[str, str]]:
    from . import (
        communication,
        config,
        discord,
        execution,
        handler,
        handler_ext,
        lifecycle,
        memory,
        migrate,
        misc,
        misc_machine,
        misc_routes,
        room_manager,
        server,
        supervisor,
        tooling,
        tooling_schema,
        tooling_schema_ext,
    )

    merged: dict[str, dict[str, str]] = {}
    for mod in (
        handler,
        handler_ext,
        tooling,
        tooling_schema,
        tooling_schema_ext,
        memory,
        execution,
        config,
        lifecycle,
        communication,
        discord,
        supervisor,
        server,
        room_manager,
        misc,
        misc_routes,
        misc_machine,
        migrate,
    ):
        merged.update(mod.STRINGS)
    return merged
