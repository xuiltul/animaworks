#!/usr/bin/env python
"""Preflight: import every core/ and server/ module and report breakage.

Catches the failure mode that took the fleet down on 2026-09-21: a branch
integration (cherry-pick/merge) that lands a *caller* without its *callee*,
so `python -c "import x"` on one entrypoint passes while every anima runner
dies at startup with ImportError / ModuleNotFoundError.

Usage (MUST run before restarting the fleet on a new branch):
    .venv/bin/python scripts/preflight_imports.py
Exit code 0 = safe to restart. Non-zero = do not restart.
"""
from __future__ import annotations

import importlib
import pathlib
import sys

FATAL = (ImportError, ModuleNotFoundError, AttributeError, NameError)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    mods: list[str] = []
    for pattern in ("core/**/*.py", "server/**/*.py"):
        for p in sorted(root.glob(pattern)):
            if "__pycache__" in p.parts:
                continue
            m = ".".join(p.relative_to(root).with_suffix("").parts)
            if m.endswith(".__init__"):
                m = m[: -len(".__init__")]
            mods.append(m)

    broken: list[tuple[str, str, str]] = []
    for m in mods:
        try:
            importlib.import_module(m)
        except FATAL as exc:
            broken.append((m, type(exc).__name__, str(exc)[:200]))
        except Exception:
            # Runtime/config errors at import time are out of scope here.
            pass

    print(f"preflight: checked={len(mods)} broken={len(broken)}")
    for m, t, e in broken:
        print(f"  BROKEN {m}: {t}: {e}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
