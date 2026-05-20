from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct ChromaDB access guard for worker-only production routing."""

import os

DIRECT_CHROMA_ENV = "ANIMAWORKS_ALLOW_DIRECT_CHROMA"


def direct_chroma_allowed() -> bool:
    """Return whether this process may instantiate native ChromaDB."""
    return os.environ.get(DIRECT_CHROMA_ENV) == "1"


def require_direct_chroma_allowed() -> None:
    """Raise when native ChromaDB is requested outside the vector worker."""
    if direct_chroma_allowed():
        return
    raise RuntimeError(
        "Direct ChromaDB access is disabled outside the vector worker. "
        f"Set {DIRECT_CHROMA_ENV}=1 only for vector-worker or explicit test processes."
    )


def enable_direct_chroma_for_process() -> None:
    """Mark the current process as an allowed native ChromaDB owner."""
    os.environ[DIRECT_CHROMA_ENV] = "1"
