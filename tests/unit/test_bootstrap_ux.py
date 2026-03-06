"""Tests for bootstrap UX improvements and Docker dev environment."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

STATIC_DIR = Path(__file__).resolve().parents[2] / "server" / "static"
DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"

BOOTSTRAP_I18N_KEYS = [
    "chat.anima_starting",
    "chat.bootstrap_complete",
    "chat.bootstrap_failed",
    "chat.bootstrap_max_retries",
    "chat.bootstrap_step_avatar",
    "chat.bootstrap_step_finish",
    "chat.bootstrap_step_identity",
    "chat.bootstrap_step_intro",
    "chat.bootstrap_step_team",
    "chat.bootstrap_step_workspace",
]

WS_BOOTSTRAP_KEYS = [
    "websocket.bootstrap_max_retries",
]


class TestBootstrapI18n:
    """Frontend i18n JSON files contain all bootstrap step keys."""

    @pytest.fixture(params=["ja", "en"])
    def locale_data(self, request):
        path = STATIC_DIR / "i18n" / f"{request.param}.json"
        with open(path, encoding="utf-8") as f:
            return request.param, json.load(f)

    def test_bootstrap_step_keys_present(self, locale_data):
        locale, data = locale_data
        missing = [k for k in BOOTSTRAP_I18N_KEYS if k not in data]
        assert not missing, f"[{locale}] Missing bootstrap keys: {missing}"

    def test_websocket_bootstrap_keys_present(self, locale_data):
        locale, data = locale_data
        missing = [k for k in WS_BOOTSTRAP_KEYS if k not in data]
        assert not missing, f"[{locale}] Missing WS bootstrap keys: {missing}"

    def test_bootstrap_values_non_empty(self, locale_data):
        locale, data = locale_data
        empty = [k for k in BOOTSTRAP_I18N_KEYS if k in data and not data[k].strip()]
        assert not empty, f"[{locale}] Empty bootstrap values: {empty}"

    def test_both_locales_have_same_bootstrap_keys(self):
        ja_path = STATIC_DIR / "i18n" / "ja.json"
        en_path = STATIC_DIR / "i18n" / "en.json"
        with open(ja_path, encoding="utf-8") as f:
            ja = json.load(f)
        with open(en_path, encoding="utf-8") as f:
            en = json.load(f)
        for key in BOOTSTRAP_I18N_KEYS + WS_BOOTSTRAP_KEYS:
            assert key in ja, f"Missing in ja.json: {key}"
            assert key in en, f"Missing in en.json: {key}"


class TestDockerComposeDev:
    """Docker compose dev override is valid and correctly structured."""

    @pytest.fixture()
    def dev_compose(self):
        path = DEMO_DIR / "docker-compose.dev.yml"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_file_exists(self):
        assert (DEMO_DIR / "docker-compose.dev.yml").exists()

    def test_valid_yaml(self, dev_compose):
        assert isinstance(dev_compose, dict)

    def test_has_services_section(self, dev_compose):
        assert "services" in dev_compose

    def test_has_animaworks_demo_service(self, dev_compose):
        assert "animaworks-demo" in dev_compose["services"]

    def test_volumes_include_static_mount(self, dev_compose):
        svc = dev_compose["services"]["animaworks-demo"]
        assert "volumes" in svc
        volumes = svc["volumes"]
        static_mount = [v for v in volumes if "server/static" in str(v)]
        assert len(static_mount) >= 1, "Missing ../server/static volume mount"

    def test_volumes_include_data_volume(self, dev_compose):
        svc = dev_compose["services"]["animaworks-demo"]
        volumes = svc["volumes"]
        data_mount = [v for v in volumes if "animaworks-demo-data" in str(v)]
        assert len(data_mount) >= 1, "Missing animaworks-demo-data volume"


class TestBootstrapCssClasses:
    """CSS file contains required bootstrap progress classes."""

    @pytest.fixture()
    def css_content(self):
        path = STATIC_DIR / "styles" / "chat.css"
        return path.read_text(encoding="utf-8")

    REQUIRED_CLASSES = [
        ".bootstrap-progress",
        ".bootstrap-progress-avatar",
        ".bootstrap-progress-avatar-img",
        ".bootstrap-progress-avatar-initial",
        ".bootstrap-progress-spinner",
        ".bootstrap-progress-step",
        ".bootstrap-step-fade",
        ".bootstrap-progress--complete",
        ".bootstrap-progress--error",
    ]

    @pytest.mark.parametrize("cls", REQUIRED_CLASSES)
    def test_css_class_defined(self, css_content, cls):
        assert cls in css_content, f"CSS class {cls} not found in chat.css"


class TestWebsocketOnEvent:
    """websocket.js exports onEvent and dispatches to handlers."""

    @pytest.fixture()
    def ws_content(self):
        path = STATIC_DIR / "modules" / "websocket.js"
        return path.read_text(encoding="utf-8")

    def test_exports_on_event(self, ws_content):
        assert "export function onEvent" in ws_content

    def test_event_handlers_map(self, ws_content):
        assert "_eventHandlers" in ws_content

    def test_max_retries_exceeded_handling(self, ws_content):
        assert "max_retries_exceeded" in ws_content

    def test_dispatches_to_handlers(self, ws_content):
        assert "_eventHandlers.get(eventType)" in ws_content


class TestPaneHostBootstrapIntegration:
    """pane-host.js subscribes to bootstrap events."""

    @pytest.fixture()
    def pane_host_content(self):
        path = STATIC_DIR / "pages" / "chat" / "pane-host.js"
        return path.read_text(encoding="utf-8")

    def test_imports_on_event(self, pane_host_content):
        assert 'import { onEvent }' in pane_host_content

    def test_subscribes_to_bootstrap(self, pane_host_content):
        assert 'onEvent("anima.bootstrap"' in pane_host_content

    def test_handles_completed(self, pane_host_content):
        assert "showBootstrapComplete" in pane_host_content

    def test_handles_max_retries(self, pane_host_content):
        assert "max_retries_exceeded" in pane_host_content

    def test_unsub_stored_in_intervals(self, pane_host_content):
        assert "pane.intervals.push(unsubBootstrap)" in pane_host_content

    def test_destroy_handles_function_entries(self, pane_host_content):
        assert 'typeof entry === "function"' in pane_host_content

    def test_clears_bootstrap_interval_on_destroy(self, pane_host_content):
        assert "clearBootstrapInterval()" in pane_host_content


class TestChatRendererBootstrap:
    """chat-renderer.js implements bootstrap progress rendering."""

    @pytest.fixture()
    def renderer_content(self):
        path = STATIC_DIR / "pages" / "chat" / "chat-renderer.js"
        return path.read_text(encoding="utf-8")

    def test_bootstrap_steps_defined(self, renderer_content):
        assert "BOOTSTRAP_STEPS" in renderer_content

    def test_render_bootstrap_progress(self, renderer_content):
        assert "function renderBootstrapProgress" in renderer_content

    def test_show_bootstrap_complete(self, renderer_content):
        assert "function showBootstrapComplete" in renderer_content

    def test_avatar_html_helper(self, renderer_content):
        assert "function _avatarHtml" in renderer_content

    def test_clear_bootstrap_interval(self, renderer_content):
        assert "function _clearBootstrapInterval" in renderer_content

    def test_exports_bootstrap_functions(self, renderer_content):
        assert "showBootstrapComplete," in renderer_content
        assert "clearBootstrapInterval:" in renderer_content

    def test_status_check_before_empty_messages(self, renderer_content):
        lines = renderer_content.split("\n")
        bootstrap_check_idx = None
        empty_check_idx = None
        for i, line in enumerate(lines):
            if "animaStatus === \"bootstrapping\"" in line:
                bootstrap_check_idx = i
            if "hs.sessions.length === 0 && history.length === 0" in line:
                if empty_check_idx is None:
                    empty_check_idx = i
        assert bootstrap_check_idx is not None, "Bootstrap status check not found"
        assert empty_check_idx is not None, "Empty messages check not found"
        assert bootstrap_check_idx < empty_check_idx, (
            "Bootstrap check must come before empty messages check"
        )

    def test_six_bootstrap_steps(self, renderer_content):
        assert "bootstrap_step_identity" in renderer_content
        assert "bootstrap_step_workspace" in renderer_content
        assert "bootstrap_step_avatar" in renderer_content
        assert "bootstrap_step_intro" in renderer_content
        assert "bootstrap_step_team" in renderer_content
        assert "bootstrap_step_finish" in renderer_content
