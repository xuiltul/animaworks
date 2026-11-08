# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from datetime import timedelta

from cli._anima_tool import current_anima_dir


def _refuse_impersonation(anima_name: str, requested_name: str) -> None:
    from core.i18n import t

    print(t("cli.board_impersonation_refused", anima=anima_name, requested=requested_name), file=sys.stderr)
    raise SystemExit(1)


def cmd_timeline_add(args) -> None:
    """Append a done/decided item to the company timeline."""
    anima_dir = current_anima_dir()
    requested_name = getattr(args, "from_name", None)
    if anima_dir is not None and requested_name and requested_name != anima_dir.name:
        _refuse_impersonation(anima_dir.name, requested_name)

    from_name = anima_dir.name if anima_dir is not None else (requested_name or "operator")
    from core.host_api import response_detail
    from core.i18n import t
    from core.infra.runtime_init import ensure_runtime_dir
    from core.internal_api import host_api

    ensure_runtime_dir()

    try:
        response = host_api.post(
            "/api/internal/timeline/add",
            json={
                "from_anima": from_name,
                "kind": args.kind,
                "text": args.text,
                "about": getattr(args, "about", None),
                "ref": getattr(args, "ref", None),
            },
            timeout=30.0,
        )
    except Exception as exc:
        print(t("cli.timeline_add_failed", detail=str(exc)), file=sys.stderr)
        raise SystemExit(1) from exc

    if response.status_code < 200 or response.status_code >= 300:
        print(t("cli.timeline_add_failed", detail=response_detail(response)), file=sys.stderr)
        raise SystemExit(1)
    print(t("cli.timeline_add_success", kind=args.kind))


def cmd_timeline_read(args) -> None:
    """Read the company timeline for the last 24 hours."""
    from core.host_api import response_detail
    from core.i18n import t
    from core.infra.runtime_init import ensure_runtime_dir
    from core.internal_api import host_api
    from core.time_utils import now_local

    ensure_runtime_dir()
    since = (now_local() - timedelta(hours=24)).isoformat()
    try:
        response = host_api.get(
            "/api/internal/timeline",
            params={"limit": getattr(args, "limit", 20), "since": since},
            timeout=30.0,
        )
    except Exception as exc:
        print(t("cli.timeline_read_failed", detail=str(exc)), file=sys.stderr)
        raise SystemExit(1) from exc

    if response.status_code < 200 or response.status_code >= 300:
        print(t("cli.timeline_read_failed", detail=response_detail(response)), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
