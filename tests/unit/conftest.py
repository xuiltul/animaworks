# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit-test fixtures: global permissions cache for ToolHandler security tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core.config.global_permissions import GlobalPermissionsCache
from core.paths import TEMPLATES_DIR


@pytest.fixture(autouse=True)
def _reset_llm_rate_guard_singleton(tmp_path: Path) -> None:
    """Point the process-wide LLM rate guard at a per-test temp file."""
    from core.config.schemas import LlmRateGuardConfig
    from core.llm.guard import rate_guard

    rate_guard._shared_guard = rate_guard.LlmRateGuard(
        config=LlmRateGuardConfig(),
        path=tmp_path / "_llm_rate_guard.json",
    )
    yield
    rate_guard._shared_guard = None


@pytest.fixture(autouse=True)
def _reset_config_caches_for_unit_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolate runtime config and event exporters from the developer machine."""
    from core.config import invalidate_cache, invalidate_vault_cache
    from core.infra.event_export import reset_event_exporters

    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path / "_runtime"))
    reset_event_exporters()
    invalidate_cache()
    invalidate_vault_cache()
    yield
    reset_event_exporters()
    invalidate_cache()
    invalidate_vault_cache()


@pytest.fixture(autouse=True)
def _block_real_llm_sdk_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any unit test that would open a real Claude Agent SDK or Codex SDK session.

    Helper calls on an Anima's own model may fall back to the Agent SDK, which
    on a developer machine would reach the real Claude login. Tests that stub
    ``claude_agent_sdk`` / ``openai_codex`` in ``sys.modules`` are unaffected.
    """
    import importlib

    def _real_class(module_name: str, class_name: str) -> type | None:
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except Exception:
            return None
        return cls if isinstance(cls, type) else None

    async def _blocked_connect(self, *args, **kwargs):
        raise RuntimeError("unit tests must not open a real Claude Agent SDK session")

    def _blocked_init(self, *args, **kwargs):
        raise RuntimeError("unit tests must not open a real Codex SDK session")

    sdk_client = _real_class("claude_agent_sdk", "ClaudeSDKClient")
    if sdk_client is not None:
        monkeypatch.setattr(sdk_client, "connect", _blocked_connect)
    codex_client = _real_class("openai_codex", "AsyncCodex")
    if codex_client is not None:
        monkeypatch.setattr(codex_client, "__init__", _blocked_init)


@pytest.fixture(autouse=True)
def _disable_unsolicited_background_reviews(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent unrelated unit tests from launching background LLM calls."""
    from core.memory.maintenance import background_review

    monkeypatch.setattr(background_review, "_review_config", lambda _anima_dir: None)


@pytest.fixture(autouse=True)
def _disable_skill_dense_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bypass the real embedding model in unit tests.

    ``core.skills.dense`` normally invokes the local HTTP embedding model;
    unit tests replace it so no real model is ever loaded.  Individual tests
    (e.g. test_skill_dense.py) override ``generate_embeddings`` themselves.
    """
    import core.skills.dense as skill_dense

    monkeypatch.setattr(skill_dense, "generate_embeddings", lambda *a, **k: [])
    skill_dense._PROCESS_CACHE.clear()
    skill_dense._QUERY_LRU.clear()
    yield
    skill_dense._PROCESS_CACHE.clear()
    skill_dense._QUERY_LRU.clear()


@pytest.fixture(autouse=True)
def _global_permissions_for_unit_tests(tmp_path: Path) -> None:
    """Load ``permissions.global.json`` template so command block patterns match production.

    Uses a dedicated subdirectory so the hash file (``run/``) does not
    pollute the test's ``tmp_path`` root.
    """
    GlobalPermissionsCache.reset()
    gp_dir = tmp_path / "_global_perms"
    gp_dir.mkdir(exist_ok=True)
    src = TEMPLATES_DIR / "_shared" / "config_defaults" / "permissions.global.json"
    dst = gp_dir / "permissions.global.json"
    shutil.copy(src, dst)
    GlobalPermissionsCache.get().load(dst, interactive=False)
    yield
    GlobalPermissionsCache.reset()


# ── Temporary root-ownership shim (remove with the old RAG branches in R07) ──
import json as _json_topology

import pytest as _pytest_topology

import core.config.resolver as _resolver_module


@pytest.fixture(autouse=True)
def _bypass_internal_api_auth_for_existing_route_tests(monkeypatch):
    """Existing internal-route tests call the router without an auth header.

    R04-1 turned every ``/api/internal/*`` route into a dependency on
    ``require_internal_caller`` (enforce by default).  To keep those tests'
    expectations unchanged we make the dependency a pass-through (returns
    None) here.  The real enforcement behaviour is exercised directly in
    ``tests/unit/server/test_internal_auth.py`` using the unpatched
    function.
    """
    # The dependency name is resolved to this module attribute by
    # create_internal_router() at router-build time.  A zero-argument
    # pass-through keeps FastAPI from treating any parameter as a query
    # dependency.
    import server.internal_auth as _internal_auth

    def _noop_internal_caller() -> None:
        return None

    monkeypatch.setattr(_internal_auth, "require_internal_caller", _noop_internal_caller)


@_pytest_topology.fixture(autouse=True)
def _legacy_topology_for_fixtureless_animas(monkeypatch):
    from pathlib import Path as _Path

    def _is_root_memory_owner(anima_dir):
        status_path = _Path(anima_dir) / "status.json"
        if not status_path.is_file():
            return False
        try:
            status = _json_topology.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, _json_topology.JSONDecodeError):
            return False
        return isinstance(status, dict) and status.get("process_model", "phase3") == "phase3"

    monkeypatch.setattr(_resolver_module, "is_root_memory_owner", _is_root_memory_owner)
    yield
