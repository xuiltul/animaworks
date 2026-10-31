from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from core.config.models import AnimaModelConfig, AnimaWorksConfig, save_config
from core.paths import get_animas_dir
from server.internal_auth import InternalCaller
from server.routes.internal import create_internal_router


def _write_anima(name: str, *, supervisor: str | None) -> Path:
    anima_dir = get_animas_dir() / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "IDENTITY.md").write_text(f"{name}\n", encoding="utf-8")
    (anima_dir / "status.json").write_text(json.dumps({"supervisor": supervisor, "enabled": True}), encoding="utf-8")
    (anima_dir / "permissions.json").write_text(json.dumps({"version": 1, "file_roots": []}), encoding="utf-8")
    return anima_dir


def _app(caller: InternalCaller) -> FastAPI:
    app = FastAPI()
    app.include_router(create_internal_router(), prefix="/api")

    @app.middleware("http")
    async def set_test_caller(request, call_next):
        request.state.internal_caller = caller
        return await call_next(request)

    return app


@pytest.mark.asyncio
async def test_control_api_uses_authenticated_ancestor_for_permission(tmp_path: Path) -> None:
    from core.config import invalidate_cache

    data_dir = get_animas_dir().parent
    data_dir.mkdir(parents=True, exist_ok=True)
    manager_dir = _write_anima("manager", supervisor=None)
    worker_dir = _write_anima("worker", supervisor="manager")
    outsider_dir = _write_anima("outsider", supervisor=None)
    save_config(
        AnimaWorksConfig(
            animas={
                "manager": AnimaModelConfig(supervisor=None),
                "worker": AnimaModelConfig(supervisor="manager"),
                "outsider": AnimaModelConfig(supervisor=None),
            }
        )
    )
    invalidate_cache()

    app = _app(InternalCaller(kind="anima", name="manager"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/internal/animas/worker/control",
            json={"action": "disable"},
        )
    assert response.status_code == 200
    assert response.json()["changed"] is True
    assert json.loads((worker_dir / "status.json").read_text(encoding="utf-8"))["enabled"] is False

    app = _app(InternalCaller(kind="anima", name="outsider"))
    before = (worker_dir / "status.json").read_bytes()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        denied = await client.post(
            "/api/internal/animas/worker/control",
            json={"action": "enable"},
        )
    assert denied.status_code == 403
    assert (worker_dir / "status.json").read_bytes() == before
    assert manager_dir.is_dir() and outsider_dir.is_dir()


@pytest.mark.asyncio
async def test_prompt_settings_api_is_limited_to_bootstrap_self_and_ancestor_injection() -> None:
    from core.anima.bootstrap_state import STATE_PENDING_USER_INPUT, write_bootstrap_state
    from core.config import invalidate_cache

    manager_dir = _write_anima("manager", supervisor=None)
    worker_dir = _write_anima("worker", supervisor="manager")
    (worker_dir / "BOOTSTRAP.md").write_text("initial setup\n", encoding="utf-8")
    write_bootstrap_state(worker_dir, {"state": STATE_PENDING_USER_INPUT})
    save_config(
        AnimaWorksConfig(
            animas={
                "manager": AnimaModelConfig(supervisor=None),
                "worker": AnimaModelConfig(supervisor="manager"),
            }
        )
    )
    invalidate_cache()

    self_app = _app(InternalCaller(kind="anima", name="worker"))
    async with AsyncClient(transport=ASGITransport(app=self_app), base_url="http://test") as client:
        self_update = await client.post(
            "/api/internal/animas/worker/prompt-settings",
            json={"setting": "identity", "content": "Human-approved identity"},
        )
    assert self_update.status_code == 200
    assert (worker_dir / "IDENTITY.md").read_text(encoding="utf-8") == "Human-approved identity"
    (worker_dir / "BOOTSTRAP.md").unlink()
    write_bootstrap_state(worker_dir, {"state": "completed"})
    async with AsyncClient(transport=ASGITransport(app=self_app), base_url="http://test") as client:
        post_bootstrap = await client.post(
            "/api/internal/animas/worker/prompt-settings",
            json={"setting": "identity", "content": "unauthorized identity change"},
        )
    assert post_bootstrap.status_code == 403
    assert (worker_dir / "IDENTITY.md").read_text(encoding="utf-8") == "Human-approved identity"
    async with AsyncClient(transport=ASGITransport(app=self_app), base_url="http://test") as client:
        own_injection = await client.post(
            "/api/internal/animas/worker/prompt-settings",
            json={"setting": "injection", "content": "Self-maintained guidance"},
        )
    assert own_injection.status_code == 200
    assert (worker_dir / "AGENTS.md").read_text(encoding="utf-8") == "Self-maintained guidance"

    supervisor_app = _app(InternalCaller(kind="anima", name="manager"))
    async with AsyncClient(transport=ASGITransport(app=supervisor_app), base_url="http://test") as client:
        subordinate_update = await client.post(
            "/api/internal/animas/worker/prompt-settings",
            json={"setting": "injection", "content": "Updated task guidance"},
        )
    assert subordinate_update.status_code == 200
    assert (worker_dir / "AGENTS.md").read_text(encoding="utf-8") == "Updated task guidance"

    outsider_app = _app(InternalCaller(kind="anima", name="outsider"))
    async with AsyncClient(transport=ASGITransport(app=outsider_app), base_url="http://test") as client:
        denied = await client.post(
            "/api/internal/animas/worker/prompt-settings",
            json={"setting": "injection", "content": "unauthorized"},
        )
    assert denied.status_code == 403
    assert manager_dir.is_dir()


@pytest.mark.asyncio
async def test_control_api_applies_model_updates_through_root_writer(tmp_path: Path) -> None:
    from core.config import invalidate_cache

    manager_dir = _write_anima("manager", supervisor=None)
    worker_dir = _write_anima("worker", supervisor="manager")
    save_config(
        AnimaWorksConfig(
            animas={
                "manager": AnimaModelConfig(supervisor=None),
                "worker": AnimaModelConfig(supervisor="manager"),
            }
        )
    )
    invalidate_cache()
    app = _app(InternalCaller(kind="anima", name="manager"))
    app.state.supervisor = MagicMock()
    app.state.supervisor.processes = {"worker": MagicMock()}
    app.state.supervisor.send_request = AsyncMock(return_value={})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        model_response = await client.post(
            "/api/internal/animas/worker/control",
            json={"action": "set_model", "model": "claude-sonnet-4-6"},
        )
        background_response = await client.post(
            "/api/internal/animas/worker/control",
            json={"action": "set_background_model", "background_model": "claude-haiku-4-5"},
        )

    assert model_response.status_code == 200
    assert background_response.status_code == 200
    status = json.loads((worker_dir / "status.json").read_text(encoding="utf-8"))
    assert status["model"] == "claude-sonnet-4-6"
    assert status["background_model"] == "claude-haiku-4-5"
    app.state.supervisor.send_request.assert_has_awaits(
        [
            call("worker", "reload_config", {}, timeout=10.0),
            call("worker", "reload_config", {}, timeout=10.0),
        ]
    )
    assert manager_dir.is_dir()


@pytest.mark.asyncio
async def test_workspace_grant_api_persists_global_and_per_anima_settings(tmp_path: Path) -> None:
    from core.config import invalidate_cache

    owner_dir = _write_anima("owner", supervisor=None)
    worker_dir = _write_anima("worker", supervisor="owner")
    save_config(
        AnimaWorksConfig(
            animas={
                "owner": AnimaModelConfig(supervisor=None),
                "worker": AnimaModelConfig(supervisor="owner"),
            }
        )
    )
    invalidate_cache()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    app = _app(InternalCaller(kind="anima", name="owner"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/internal/workspace/grant",
            json={
                "alias": "project",
                "path": str(workspace),
                "target_anima": "worker",
                "make_default": True,
                "human_origin": True,
            },
        )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["permissions_changed"] is True
    assert (
        str(workspace.resolve())
        in json.loads((worker_dir / "permissions.json").read_text(encoding="utf-8"))["file_roots"]
    )
    status = json.loads((worker_dir / "status.json").read_text(encoding="utf-8"))
    assert status["default_workspace"] == result["qualified_alias"]
    from core.config.models import load_config

    assert load_config().workspaces["project"] == str(workspace.resolve())
    assert owner_dir.is_dir()


@pytest.mark.asyncio
async def test_company_assignment_internal_api_is_operator_only() -> None:
    app = _app(InternalCaller(kind="operator", name="operator"))
    with patch("core.org.company.assign_animas", return_value=["ASSIGN alice -> alpha"]) as assign:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/internal/company/assign",
                json={"anima_names": ["alice"], "company_name": "alpha", "unassign": False},
            )
    assert response.status_code == 200
    assert response.json()["lines"] == ["ASSIGN alice -> alpha"]
    assign.assert_called_once()

    app = _app(InternalCaller(kind="anima", name="alice"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        denied = await client.post(
            "/api/internal/company/assign",
            json={"anima_names": ["alice"], "company_name": "alpha", "unassign": False},
        )
    assert denied.status_code == 403
