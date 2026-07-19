from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve read-only company knowledge and skill resources for an Anima."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompanyResources:
    """Company-scoped resource directories visible to one assigned Anima."""

    name: str
    root: Path
    knowledge_dir: Path
    skills_dir: Path


def infer_data_dir(anima_dir: Path) -> Path:
    """Infer the runtime data directory without consulting unrelated runtimes."""
    anima_dir = Path(anima_dir)
    if anima_dir.parent.name == "animas":
        return anima_dir.parent.parent

    from core.paths import get_data_dir

    return get_data_dir()


def get_company_resources(
    anima_dir: Path,
    *,
    data_dir: Path | None = None,
) -> CompanyResources | None:
    """Return the assigned company's resource roots, or ``None`` for legacy Animas.

    Company names are required to be a single safe path component.  This keeps a
    malformed ``status.json`` from turning resource discovery into an arbitrary
    filesystem read.
    """
    from core.config.models import read_anima_company

    company = read_anima_company(Path(anima_dir))
    if not company or company in {".", ".."} or Path(company).name != company or "\\" in company:
        return None

    base = Path(data_dir) if data_dir is not None else infer_data_dir(Path(anima_dir))
    companies_dir = (base / "companies").resolve()
    root = (companies_dir / company).resolve()
    try:
        root.relative_to(companies_dir)
    except ValueError:
        return None

    return CompanyResources(
        name=company,
        root=root,
        knowledge_dir=root / "knowledge",
        skills_dir=root / "skills",
    )


def company_resource_pointer(path: Path) -> str | None:
    """Return a canonical ``companies/...`` pointer for a company resource."""
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if part != "companies" or len(parts) <= index + 3:
            continue
        if parts[index + 2] not in {"knowledge", "skills"}:
            continue
        return str(Path(*parts[index:]))
    return None
