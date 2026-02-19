# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import sys


# ── Create Anima ─────────────────────────────────────────


def cmd_create_anima(args: argparse.Namespace) -> None:
    """Create a new Digital Anima."""
    from pathlib import Path

    from core.init import ensure_runtime_dir
    from core.paths import get_data_dir, get_animas_dir
    from core.anima_factory import (
        create_blank,
        create_from_md,
        create_from_template,
        validate_anima_name,
    )

    from cli.commands.init_cmd import _register_anima_in_config

    ensure_runtime_dir(skip_animas=True)
    data_dir = get_data_dir()
    animas_dir = get_animas_dir()
    animas_dir.mkdir(parents=True, exist_ok=True)

    supervisor = getattr(args, "supervisor", None)

    if args.from_md:
        md_path = Path(args.from_md).resolve()
        role = getattr(args, "role", None)
        anima_dir = create_from_md(
            animas_dir, md_path, name=args.name, supervisor=supervisor, role=role,
        )
        _register_anima_in_config(data_dir, anima_dir.name)
        print(f"Created anima '{anima_dir.name}' from {md_path.name}")
        return

    if args.template:
        anima_dir = create_from_template(
            animas_dir, args.template, anima_name=args.name
        )
        _register_anima_in_config(data_dir, anima_dir.name)
        print(f"Created anima '{anima_dir.name}' from template '{args.template}'")
        return

    # Default: blank creation
    name = args.name
    if not name:
        print("Error: --name is required for blank anima creation")
        sys.exit(1)
    err = validate_anima_name(name)
    if err:
        print(f"Error: {err}")
        sys.exit(1)
    anima_dir = create_blank(animas_dir, name)
    _register_anima_in_config(data_dir, anima_dir.name)
    print(f"Created blank anima '{anima_dir.name}'")


# ── Chat ───────────────────────────────────────────────────


def cmd_chat(args: argparse.Namespace) -> None:
    """Chat with an anima (via gateway or direct)."""
    if args.local:
        from core.init import ensure_runtime_dir
        from core.paths import get_animas_dir, get_shared_dir
        from core.anima import DigitalAnima

        ensure_runtime_dir()
        anima_dir = get_animas_dir() / args.anima
        if not anima_dir.exists():
            print(f"Anima not found: {args.anima}")
            sys.exit(1)

        anima = DigitalAnima(anima_dir, get_shared_dir())
        response = asyncio.run(
            anima.process_message(args.message, from_person=args.from_person)
        )
        print(response)
    else:
        from cli._gateway import gateway_request

        data = gateway_request(
            args,
            "POST",
            f"/api/animas/{args.anima}/chat",
            json={"message": args.message, "from_person": args.from_person},
            timeout=300.0,
        )
        print(data.get("response", data.get("error", "Unknown error")))


# ── Heartbeat ──────────────────────────────────────────────


def cmd_heartbeat(args: argparse.Namespace) -> None:
    """Trigger heartbeat (via gateway or direct)."""
    if args.local:
        from core.init import ensure_runtime_dir
        from core.paths import get_animas_dir, get_shared_dir
        from core.anima import DigitalAnima

        ensure_runtime_dir()
        anima_dir = get_animas_dir() / args.anima
        if not anima_dir.exists():
            print(f"Anima not found: {args.anima}")
            sys.exit(1)

        anima = DigitalAnima(anima_dir, get_shared_dir())
        result = asyncio.run(anima.run_heartbeat())
        print(f"[{result.action}] {result.summary[:500]}")
    else:
        from cli._gateway import gateway_request

        data = gateway_request(
            args,
            "POST",
            f"/api/animas/{args.anima}/trigger",
            timeout=120.0,
        )
        print(data)
