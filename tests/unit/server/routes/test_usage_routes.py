from __future__ import annotations

import io
import json
import subprocess
import urllib.error
from pathlib import Path

from server.routes import usage_routes


def _jwt(payload: dict[str, object]) -> str:
    import base64

    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode("utf-8").rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8").rstrip("=")
    return f"{header}.{body}.sig"


class _FakeResponse:
    def __init__(self, payload: dict[str, object]):
        self.status = 200
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_refresh_codex_token_updates_auth_file(tmp_path: Path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    auth_data = {
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": _jwt(
                {
                    "client_id": "client-123",
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "acct-old",
                    },
                }
            ),
            "refresh_token": "refresh-123",
            "account_id": "acct-old",
        },
    }
    auth_path.write_text(json.dumps(auth_data), encoding="utf-8")

    def fake_urlopen(req, timeout=0):
        assert req.full_url == "https://auth.openai.com/oauth/token"
        body = json.loads(req.data.decode("utf-8"))
        assert body["grant_type"] == "refresh_token"
        assert body["client_id"] == "client-123"
        assert body["refresh_token"] == "refresh-123"
        return _FakeResponse(
            {
                "access_token": _jwt(
                    {
                        "client_id": "client-123",
                        "https://api.openai.com/auth": {
                            "chatgpt_account_id": "acct-new",
                        },
                    }
                ),
                "id_token": _jwt({"aud": ["client-123"]}),
                "refresh_token": "refresh-456",
            }
        )

    monkeypatch.setattr(usage_routes.urllib.request, "urlopen", fake_urlopen)
    token, account_id = usage_routes._refresh_codex_token(auth_path, auth_data)

    saved = json.loads(auth_path.read_text("utf-8"))
    assert token == saved["tokens"]["access_token"]
    assert account_id == "acct-new"
    assert saved["tokens"]["account_id"] == "acct-new"
    assert saved["tokens"]["refresh_token"] == "refresh-456"
    assert "last_refresh" in saved


def test_fetch_openai_usage_refreshes_after_401(monkeypatch):
    old_token = _jwt(
        {
            "client_id": "client-123",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-123",
            },
        }
    )
    new_token = _jwt(
        {
            "client_id": "client-123",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-123",
            },
        }
    )

    calls: list[str] = []

    def fake_read_codex_credentials():
        if calls:
            return new_token, "acct-123"
        return old_token, "acct-123"

    def fake_urlopen(req, timeout=0):
        calls.append(req.headers.get("Authorization", ""))
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                req.full_url,
                401,
                "Unauthorized",
                hdrs=None,
                fp=io.BytesIO(b'{"error":{"code":"token_expired"}}'),
            )
        return _FakeResponse(
            {
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 12,
                        "reset_at": 1775000000,
                        "limit_window_seconds": 18000,
                    },
                    "secondary_window": {
                        "used_percent": 34,
                        "reset_at": 1775400000,
                        "limit_window_seconds": 604800,
                    },
                }
            }
        )

    monkeypatch.setattr(usage_routes, "_CACHE", {})
    monkeypatch.setattr(usage_routes, "_read_codex_credentials", fake_read_codex_credentials)
    monkeypatch.setattr(usage_routes, "_read_codex_auth_data", lambda: (Path("auth.json"), {"tokens": {}}))
    monkeypatch.setattr(usage_routes, "_refresh_codex_token", lambda path, data: (new_token, "acct-123"))
    monkeypatch.setattr(usage_routes.urllib.request, "urlopen", fake_urlopen)

    result = usage_routes._fetch_openai_usage(skip_cache=True)

    assert result["provider"] == "openai"
    assert result["5h"]["remaining"] == 88
    assert result["Week"]["remaining"] == 66
    assert len(calls) == 2


# ── macOS Keychain tests ──────────────────────────────────────────────────


def _keychain_json(access_token: str = "sk-test", refresh_token: str = "rt-test", expires_at: int = 9999999999999) -> str:
    return json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": access_token,
                "refreshToken": refresh_token,
                "expiresAt": expires_at,
            }
        }
    )


def _fake_security_run(keychain_json: str):
    """Return a callable that simulates ``security find-generic-password -w``."""

    def _run(cmd, *, capture_output=False, text=False, timeout=None):
        result = subprocess.CompletedProcess(cmd, 0, stdout=keychain_json + "\n", stderr="")
        return result

    return _run


def test_read_keychain_claude_credential_returns_token(monkeypatch):
    kc_json = _keychain_json("sk-abc", "rt-xyz", 1700000000000)
    monkeypatch.setattr(usage_routes.sys, "platform", "darwin")
    monkeypatch.setattr(usage_routes.subprocess, "run", _fake_security_run(kc_json))

    token, refresh, expires = usage_routes._read_keychain_claude_credential()

    assert token == "sk-abc"
    assert refresh == "rt-xyz"
    assert expires == 1700000000000


def test_read_keychain_credential_returns_none_on_non_darwin(monkeypatch):
    monkeypatch.setattr(usage_routes.sys, "platform", "linux")

    token, refresh, expires = usage_routes._read_keychain_claude_credential()

    assert token is None
    assert refresh is None
    assert expires == 0


def test_read_keychain_credential_returns_none_on_security_failure(monkeypatch):
    monkeypatch.setattr(usage_routes.sys, "platform", "darwin")

    def _failing_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not found")

    monkeypatch.setattr(usage_routes.subprocess, "run", _failing_run)

    token, refresh, expires = usage_routes._read_keychain_claude_credential()

    assert token is None


def test_select_best_claude_credential_falls_back_to_keychain(monkeypatch):
    """When no .credentials.json exists, Keychain provides the token."""
    # Make file discovery return nothing
    monkeypatch.setattr(usage_routes, "_discover_claude_cred_paths", lambda: [])
    # Provide Keychain credentials
    kc_json = _keychain_json("sk-kc", "rt-kc", 2000000000000)
    monkeypatch.setattr(usage_routes.sys, "platform", "darwin")
    monkeypatch.setattr(usage_routes.subprocess, "run", _fake_security_run(kc_json))

    path, token, refresh, expires = usage_routes._select_best_claude_credential()

    assert path is None  # Keychain source indicated by None path
    assert token == "sk-kc"
    assert refresh == "rt-kc"
    assert expires == 2000000000000


def test_select_best_credential_prefers_file_over_keychain(monkeypatch, tmp_path):
    """File-based credentials take priority over Keychain."""
    cred_file = tmp_path / ".credentials.json"
    cred_file.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "sk-file",
                    "refreshToken": "rt-file",
                    "expiresAt": 3000000000000,
                }
            }
        )
    )
    monkeypatch.setattr(usage_routes, "_discover_claude_cred_paths", lambda: [str(cred_file)])
    # Keychain also has credentials but should NOT be used
    kc_json = _keychain_json("sk-kc-should-not-be-used", "rt-kc", 1000000000000)
    monkeypatch.setattr(usage_routes.sys, "platform", "darwin")
    monkeypatch.setattr(usage_routes.subprocess, "run", _fake_security_run(kc_json))

    path, token, refresh, expires = usage_routes._select_best_claude_credential()

    assert path == cred_file
    assert token == "sk-file"


def test_read_claude_token_uses_keychain_refresh(monkeypatch):
    """When token is expired and from Keychain, refresh via Keychain path."""
    expired_ms = 1000  # long expired
    monkeypatch.setattr(
        usage_routes,
        "_select_best_claude_credential",
        lambda: (None, "sk-expired", "rt-kc", expired_ms),
    )
    monkeypatch.setattr(
        usage_routes,
        "_refresh_keychain_claude_token",
        lambda rt: "sk-refreshed" if rt == "rt-kc" else None,
    )

    token = usage_routes._read_claude_token()

    assert token == "sk-refreshed"


def test_fetch_claude_usage_works_with_keychain(monkeypatch):
    """End-to-end: Keychain token → successful usage fetch."""
    monkeypatch.setattr(usage_routes, "_CACHE", {})
    monkeypatch.setattr(
        usage_routes,
        "_read_claude_token",
        lambda: "sk-kc-token",
    )

    def fake_urlopen(req, timeout=0):
        assert "Bearer sk-kc-token" in req.headers.get("Authorization", "")
        return _FakeResponse(
            {
                "five_hour": {"utilization": 25, "resets_at": "2026-09-14T02:00:00+00:00"},
                "seven_day": {"utilization": 60, "resets_at": "2026-09-20T00:00:00+00:00"},
            }
        )

    monkeypatch.setattr(usage_routes.urllib.request, "urlopen", fake_urlopen)

    result = usage_routes._fetch_claude_usage(skip_cache=True)

    assert result["provider"] == "claude"
    assert result["five_hour"]["remaining"] == 75
    assert result["seven_day"]["remaining"] == 40


def test_keychain_credential_values_not_in_logs(monkeypatch, caplog):
    """Ensure credential values are never logged."""
    import logging

    kc_json = _keychain_json("sk-secret-token-value", "rt-secret-refresh", 9999999999999)
    monkeypatch.setattr(usage_routes.sys, "platform", "darwin")
    monkeypatch.setattr(usage_routes.subprocess, "run", _fake_security_run(kc_json))

    with caplog.at_level(logging.DEBUG, logger="animaworks.routes.usage"):
        token, _, _ = usage_routes._read_keychain_claude_credential()

    assert token == "sk-secret-token-value"
    # Credential values must not appear in log output
    for record in caplog.records:
        assert "sk-secret-token-value" not in record.getMessage()
        assert "rt-secret-refresh" not in record.getMessage()
