from __future__ import annotations

import argparse
import asyncio
import sys


# ── Create Person ─────────────────────────────────────────


def cmd_create_person(args: argparse.Namespace) -> None:
    """Create a new Digital Person."""
    from pathlib import Path

    from core.init import ensure_runtime_dir
    from core.paths import get_data_dir, get_persons_dir
    from core.person_factory import (
        create_blank,
        create_from_md,
        create_from_template,
        validate_person_name,
    )

    from cli.commands.init_cmd import _register_person_in_config

    ensure_runtime_dir(skip_persons=True)
    data_dir = get_data_dir()
    persons_dir = get_persons_dir()
    persons_dir.mkdir(parents=True, exist_ok=True)

    if args.from_md:
        md_path = Path(args.from_md).resolve()
        person_dir = create_from_md(persons_dir, md_path, name=args.name)
        _register_person_in_config(data_dir, person_dir.name)
        print(f"Created person '{person_dir.name}' from {md_path.name}")
        return

    if args.template:
        person_dir = create_from_template(
            persons_dir, args.template, person_name=args.name
        )
        _register_person_in_config(data_dir, person_dir.name)
        print(f"Created person '{person_dir.name}' from template '{args.template}'")
        return

    # Default: blank creation
    name = args.name
    if not name:
        print("Error: --name is required for blank person creation")
        sys.exit(1)
    err = validate_person_name(name)
    if err:
        print(f"Error: {err}")
        sys.exit(1)
    person_dir = create_blank(persons_dir, name)
    _register_person_in_config(data_dir, person_dir.name)
    print(f"Created blank person '{person_dir.name}'")


# ── Chat ───────────────────────────────────────────────────


def cmd_chat(args: argparse.Namespace) -> None:
    """Chat with a person (via gateway or direct)."""
    if args.local:
        from core.init import ensure_runtime_dir
        from core.paths import get_persons_dir, get_shared_dir
        from core.person import DigitalPerson

        ensure_runtime_dir()
        person_dir = get_persons_dir() / args.person
        if not person_dir.exists():
            print(f"Person not found: {args.person}")
            sys.exit(1)

        person = DigitalPerson(person_dir, get_shared_dir())
        response = asyncio.run(
            person.process_message(args.message, from_person=args.from_person)
        )
        print(response)
    else:
        from cli._gateway import gateway_request

        data = gateway_request(
            args,
            "POST",
            f"/api/persons/{args.person}/chat",
            json={"message": args.message, "from_person": args.from_person},
            timeout=300.0,
        )
        print(data.get("response", data.get("error", "Unknown error")))


# ── Heartbeat ──────────────────────────────────────────────


def cmd_heartbeat(args: argparse.Namespace) -> None:
    """Trigger heartbeat (via gateway or direct)."""
    if args.local:
        from core.init import ensure_runtime_dir
        from core.paths import get_persons_dir, get_shared_dir
        from core.person import DigitalPerson

        ensure_runtime_dir()
        person_dir = get_persons_dir() / args.person
        if not person_dir.exists():
            print(f"Person not found: {args.person}")
            sys.exit(1)

        person = DigitalPerson(person_dir, get_shared_dir())
        result = asyncio.run(person.run_heartbeat())
        print(f"[{result.action}] {result.summary[:500]}")
    else:
        from cli._gateway import gateway_request

        data = gateway_request(
            args,
            "POST",
            f"/api/persons/{args.person}/trigger",
            timeout=120.0,
        )
        print(data)
