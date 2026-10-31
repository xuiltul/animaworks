# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Startup guards for enclave mode.

When ``AnimaWorksConfig.enclave.enabled`` is true, :func:`collect_enclave_violations`
walks the config and the runtime data directory and returns a human-readable list of
everything that would make the enclave unsafe.  :func:`enforce_enclave_runtime`
raises :class:`EnclaveViolationError` if there are any.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from collections.abc import Iterable
from pathlib import Path

from core.auth.manager import load_auth
from core.config.schemas import AnimaWorksConfig, load_permissions
from core.exceptions import EnclaveViolationError
from core.i18n import t

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
_AUTH_MODES_OK = {"password", "multi_user"}


def collect_enclave_violations(
    config: AnimaWorksConfig,
    data_dir: Path,
    *,
    host: str | None,
) -> list[str]:
    """Return human-readable violation strings for an enclave-enabled config.

    Returns an empty list when ``config.enclave.enabled`` is ``False``.
    """
    # ``is not True`` (rather than ``not``) so that mocks or non-real
    # values short-circuit without being treated as an enabled enclave.
    if config.enclave.enabled is not True:
        return []

    locale = config.locale
    enclave = config.enclave
    violations: list[str] = []

    # 1. Required fields and entry anima presence.
    if not enclave.name:
        violations.append(t("enclave.guard.name_required", locale=locale))
    if not enclave.socket_path:
        violations.append(t("enclave.guard.socket_path_required", locale=locale))
    if not enclave.allowed_peer_uids:
        violations.append(t("enclave.guard.peer_uids_required", locale=locale))
    try:
        from core.enclave.egress.config import load_egress_config

        load_egress_config(enclave.egress)
    except Exception:
        violations.append(t("enclave.guard.egress_invalid", locale=locale))
    if enclave.socket_group and not _group_exists(enclave.socket_group):
        violations.append(t("enclave.guard.socket_group_missing", locale=locale, group=enclave.socket_group))
    if not enclave.entry_anima:
        violations.append(t("enclave.guard.entry_anima_required", locale=locale))
    else:
        entry_dir = data_dir / "animas" / enclave.entry_anima
        if not entry_dir.is_dir():
            violations.append(t("enclave.guard.entry_anima_missing", locale=locale, anima=enclave.entry_anima))

    # 2. No external event export.
    if config.event_export.url:
        violations.append(t("enclave.guard.event_export_url", locale=locale))

    # 3. All external integrations disabled.
    ext = config.external_messaging
    for label, channel in (
        ("slack", ext.slack),
        ("chatwork", ext.chatwork),
        ("discord", ext.discord),
        ("zoom", ext.zoom),
    ):
        if getattr(channel, "enabled", False):
            violations.append(t("enclave.guard.external_enabled", locale=locale, channel=label))
    for label, cfg in (
        ("github_webhook", config.github_webhook),
        ("phone", config.phone),
        ("human_notification", config.human_notification),
        ("external_tasks", config.external_tasks),
    ):
        if getattr(cfg, "enabled", False):
            violations.append(t("enclave.guard.external_enabled", locale=locale, channel=label))

    # 4. Ownership and permissions of the data dir and every anima dir.
    for entry in _guard_dirs(data_dir):
        try:
            st = entry.stat()
        except OSError as exc:
            logger.warning("Failed to stat enclave guard dir %s: %s", entry, exc)
            continue
        if st.st_uid != os.getuid():
            violations.append(t("enclave.guard.dir_owner", locale=locale, path=str(entry)))
        if st.st_mode & 0o077:
            violations.append(t("enclave.guard.dir_mode", locale=locale, path=str(entry)))

    # 5. LLM credential allow-list.
    allowed = set(enclave.allowed_llm_credentials)
    if not allowed:
        violations.append(t("enclave.guard.llm_credentials_required", locale=locale))
    for credential in _referenced_credentials(config, data_dir, violations, locale):
        if credential and credential not in allowed:
            violations.append(t("enclave.guard.llm_credential_not_allowed", locale=locale, credential=credential))

    # 6. No anima has a "/" file root.
    animas_dir = data_dir / "animas"
    if animas_dir.is_dir():
        for anima_dir in sorted(animas_dir.iterdir()):
            if not anima_dir.is_dir():
                continue
            try:
                perms = load_permissions(anima_dir, read_only=True)
            except Exception as exc:  # noqa: BLE001 - fail closed on unreadable permission file
                logger.warning("Failed to load permissions for %s: %s", anima_dir, exc)
                violations.append(t("enclave.guard.file_root_unreadable", locale=locale, anima=anima_dir.name))
                continue
            if "/" in perms.file_roots:
                violations.append(t("enclave.guard.file_root_full_access", locale=locale, anima=anima_dir.name))

    # 7. Auth must be password/multi_user and not trust localhost.
    try:
        auth = load_auth()
    except Exception as exc:  # noqa: BLE001 - fail closed if auth can't be read
        logger.warning("Failed to load auth for enclave guard: %s", exc)
        violations.append(t("enclave.guard.auth_unreadable", locale=locale))
    else:
        if auth.auth_mode not in _AUTH_MODES_OK:
            violations.append(t("enclave.guard.auth_mode", locale=locale, mode=auth.auth_mode))
        if auth.trust_localhost:
            violations.append(t("enclave.guard.auth_trust_localhost", locale=locale))

    # 8. Bind host must be loopback when a host is specified.
    if host is not None and host not in _LOCAL_HOSTS:
        violations.append(t("enclave.guard.host", locale=locale, host=host))

    # 9. Targets the OS firewall must block are unreachable.
    for target in enclave.egress_probe:
        if _probe_reachable(target, enclave.egress_probe_timeout_s):
            violations.append(t("enclave.guard.egress_probe_reachable", locale=locale, target=target))

    return violations


def enforce_enclave_runtime(
    config: AnimaWorksConfig,
    data_dir: Path,
    *,
    host: str | None,
) -> None:
    """Raise :class:`EnclaveViolationError` if any enclave guard fails."""
    violations = collect_enclave_violations(config, data_dir, host=host)
    if violations:
        raise EnclaveViolationError("\n".join(violations))


def _probe_reachable(target: str, timeout_s: float) -> bool:
    """Return whether a TCP connection to ``HOST:PORT`` succeeds.

    A malformed target counts as reachable so a typo cannot disable the check.
    """
    host, _, port = target.rpartition(":")
    host = host.strip("[]")
    try:
        port_number = int(port)
    except ValueError:
        return True
    if not host:
        return True
    try:
        with socket.create_connection((host, port_number), timeout=timeout_s):
            return True
    except OSError:
        return False


def _group_exists(group: str) -> bool:
    """Return whether the named system group exists."""
    import grp

    try:
        grp.getgrnam(group)
        return True
    except KeyError:
        return False


def _guard_dirs(data_dir: Path) -> Iterable[Path]:
    """Yield ``data_dir`` and every anima directory for ownership checks."""
    yield data_dir
    animas_dir = data_dir / "animas"
    if animas_dir.is_dir():
        for anima_dir in sorted(animas_dir.iterdir()):
            if anima_dir.is_dir():
                yield anima_dir


def _referenced_credentials(
    config: AnimaWorksConfig,
    data_dir: Path,
    violations: list[str],
    locale: str | None,
) -> Iterable[str]:
    """Yield every non-empty LLM credential referenced by the config.

    Covers the fields listed in the enclave P1 plan plus the per-anima
    ``status.json`` ``credential`` / ``background_credential`` values.
    """
    yield config.anima_defaults.credential
    if config.anima_defaults.background_credential:
        yield config.anima_defaults.background_credential

    consol = config.consolidation
    for value in (
        consol.llm_credential,
        consol.weekly_llm_credential,
        consol.llm_fallback_credential,
        consol.fact_reconcile_credential,
        consol.live_fact_credential,
    ):
        if value:
            yield value

    if config.local_llm.credential:
        yield config.local_llm.credential

    # Per-ask model profiles resolve their credential the same way a chat
    # model override does, so they must stay inside the allow-list too.
    from core.config.model_config import _fallback_credential_name
    from core.config.model_mode import parse_fallback_entry

    for spec in config.enclave.model_profiles.values():
        parsed = parse_fallback_entry(spec, config) if spec else None
        if parsed is not None:
            yield _fallback_credential_name(parsed[1], config) or ""

    animas_dir = data_dir / "animas"
    if not animas_dir.is_dir():
        return
    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir():
            continue
        status_path = anima_dir / "status.json"
        if not status_path.is_file():
            continue
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # An unreadable status.json could hide a disallowed credential.
            logger.warning("Failed to read %s for enclave credential check: %s", status_path, exc)
            violations.append(t("enclave.guard.status_unreadable", locale=locale, anima=anima_dir.name))
            continue
        if not isinstance(data, dict):
            violations.append(t("enclave.guard.status_unreadable", locale=locale, anima=anima_dir.name))
            continue
        for key in ("credential", "background_credential"):
            value = data.get(key)
            if isinstance(value, str) and value:
                yield value
