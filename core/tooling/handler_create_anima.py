from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""CreateAnimaMixin — anima creation from character sheet."""

import json as _json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.tooling.handler_base import _error_result

if TYPE_CHECKING:
    pass

logger = logging.getLogger("animaworks.tool_handler")


def _server_base_url() -> str:
    return os.environ.get("ANIMAWORKS_SERVER_URL", "http://localhost:18500").rstrip("/")


class CreateAnimaMixin:
    """Mixin for create_anima tool handler."""

    # Declared for type-checker visibility
    _anima_dir: Path
    _anima_name: str

    def _handle_create_anima(self, args: dict[str, Any]) -> str:
        """Create a new anima from a character sheet via anima_factory."""
        from core.anima_factory import create_from_md
        from core.paths import get_animas_dir

        content = args.get("character_sheet_content")
        sheet_path_raw = args.get("character_sheet_path")
        name = args.get("name")
        explicit_supervisor = args.get("supervisor")

        if content:
            md_path = None
        elif sheet_path_raw:
            md_path = Path(sheet_path_raw).expanduser()
            if not md_path.is_absolute():
                md_path = (self._anima_dir / md_path).resolve()
                if not md_path.is_relative_to(self._anima_dir.resolve()):
                    return _error_result(
                        "PermissionDenied",
                        "character_sheet_path must be within anima directory.",
                    )
            else:
                # Absolute paths are intentionally allowed without directory
                # restriction — the CLI and human operators specify full paths.
                # create_from_md validates the content as a character sheet,
                # so passing an arbitrary file (e.g. /etc/passwd) will fail
                # schema validation rather than leaking data.
                md_path = md_path.resolve()
            if not md_path.exists():
                return _error_result(
                    "FileNotFound",
                    f"Character sheet not found: {md_path}",
                    suggestion=("Use character_sheet_content to pass content directly, or ensure the file exists"),
                )
            # Sandbox paths may not be readable by the host server; read content
            # now so EROFS fallback can pass character_sheet_content.
            try:
                content = md_path.read_text(encoding="utf-8")
            except OSError:
                # Still try direct create_from_md with path; fallback may fail
                # if we cannot load content either.
                pass
        else:
            return _error_result(
                "MissingParameter",
                "Either character_sheet_content or character_sheet_path is required",
            )

        try:
            # Prefer in-memory content when available so sandbox-local paths are
            # not re-read, and EROFS fallback can reuse the same payload.
            anima_dir = create_from_md(
                get_animas_dir(),
                md_path if not content else None,
                name=name,
                content=content,
                supervisor=explicit_supervisor,
            )
        except FileExistsError as e:
            return _error_result(
                "AnimaExists",
                str(e),
                suggestion="Choose a different name",
            )
        except ValueError as e:
            return _error_result("InvalidCharacterSheet", str(e))
        except OSError as e:
            # Sandbox EROFS/EACCES: fall back to server internal API
            return self._create_anima_via_server(
                content=content,
                name=name,
                supervisor=explicit_supervisor,
                original_error=e,
            )

        return self._finalize_create_anima(anima_dir)

    def _create_anima_via_server(
        self,
        *,
        content: str | None,
        name: str | None,
        supervisor: str | None,
        original_error: OSError,
    ) -> str:
        """Create anima via /api/internal/anima/create when local FS is read-only."""
        if not content:
            return _error_result(
                "PermissionDenied",
                (
                    f"Cannot create anima: filesystem error ({original_error}) "
                    "and character sheet content is unavailable for server fallback"
                ),
            )

        try:
            import httpx
        except ImportError:
            return _error_result(
                "PermissionDenied",
                f"Cannot create anima: filesystem error ({original_error})",
            )

        payload: dict[str, Any] = {
            "character_sheet_content": content,
            "calling_anima": self._anima_name or "",
        }
        if name:
            payload["name"] = name
        if supervisor:
            payload["supervisor"] = supervisor

        try:
            resp = httpx.post(
                f"{_server_base_url()}/api/internal/anima/create",
                json=payload,
                timeout=60.0,
            )
        except Exception as exc:
            return _error_result(
                "PermissionDenied",
                (f"Cannot create anima: filesystem error ({original_error}); server fallback unreachable ({exc})"),
            )

        if resp.status_code == 409:
            detail = _extract_detail(resp)
            return _error_result(
                "AnimaExists",
                detail,
                suggestion="Choose a different name",
            )
        if resp.status_code == 422:
            detail = _extract_detail(resp)
            return _error_result("InvalidCharacterSheet", detail)
        if resp.status_code >= 400:
            detail = _extract_detail(resp)
            return _error_result(
                "PermissionDenied",
                (
                    f"Cannot create anima: filesystem error ({original_error}); "
                    f"server fallback failed ({resp.status_code}: {detail})"
                ),
            )

        try:
            data = resp.json()
        except Exception:
            data = {}
        anima_dir_str = data.get("anima_dir", "")
        anima_name = Path(anima_dir_str).name if anima_dir_str else (name or "unknown")
        # Server already did supervisor fallback + config registration
        logger.info(
            "create_anima: created '%s' via server API (EROFS fallback) at %s",
            anima_name,
            anima_dir_str,
        )
        return f"Anima '{anima_name}' created successfully at {anima_dir_str}. Reload the server to activate."

    def _finalize_create_anima(self, anima_dir: Path) -> str:
        """Local post-create: supervisor fallback + config registration."""
        from core.paths import get_data_dir

        status_path = anima_dir / "status.json"
        if status_path.exists() and self._anima_name:
            try:
                status_data = _json.loads(status_path.read_text(encoding="utf-8"))
                if not status_data.get("supervisor"):
                    status_data["supervisor"] = self._anima_name
                    status_path.write_text(
                        _json.dumps(status_data, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    logger.debug(
                        "Set fallback supervisor '%s' for '%s'",
                        self._anima_name,
                        anima_dir.name,
                    )
            except (OSError, _json.JSONDecodeError):
                logger.warning("Failed to set fallback supervisor", exc_info=True)

        try:
            from cli.commands.init_cmd import _register_anima_in_config

            _register_anima_in_config(get_data_dir(), anima_dir.name)
        except Exception:
            logger.warning("Failed to register anima in config.json", exc_info=True)

        logger.info("create_anima: created '%s' at %s", anima_dir.name, anima_dir)
        return f"Anima '{anima_dir.name}' created successfully at {anima_dir}. Reload the server to activate."


def _extract_detail(resp: Any) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict):
            detail = data.get("detail", data)
            return str(detail)
    except Exception:
        pass
    return resp.text or f"HTTP {resp.status_code}"
