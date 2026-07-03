# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.execution.rate_guard — file-backed fail-open guard."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import core.execution.rate_guard as rate_guard
from core.config.schemas import LlmRateGuardConfig
from core.execution.rate_guard import LlmRateGuard, _load_guard_config


def _guard(tmp_path: Path, **cfg_overrides) -> LlmRateGuard:
    cfg = LlmRateGuardConfig(**cfg_overrides)
    return LlmRateGuard(config=cfg, path=tmp_path / "llm_rate_guard.json")


class TestReportAndQuery:
    def test_report_then_blocked(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path)
        guard.report_block("anthropic", 60, "rate_limit")
        remaining = guard.blocked_remaining("anthropic")
        assert 0 < remaining <= 60

    def test_unblocked_family_returns_zero(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path)
        guard.report_block("anthropic", 60, "rate_limit")
        assert guard.blocked_remaining("openai") == 0.0

    def test_expired_entry_returns_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "llm_rate_guard.json"
        path.write_text(json.dumps({"anthropic": {"blocked_until": time.time() - 5, "reason": "rate_limit"}}))
        guard = _guard(tmp_path)
        assert guard.blocked_remaining("anthropic") == 0.0

    def test_reason_and_updated_by_persisted(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ANIMAWORKS_ANIMA_NAME", "aoi")
        guard = _guard(tmp_path)
        guard.report_block("anthropic", 30, "overloaded")
        state = json.loads((tmp_path / "llm_rate_guard.json").read_text())
        assert state["anthropic"]["reason"] == "overloaded"
        assert state["anthropic"]["updated_by"] == "aoi"


class TestClamp:
    def test_huge_block_clamped_to_max(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path, max_block_seconds=600)
        guard.report_block("anthropic", 999999, "rate_limit")
        assert guard.blocked_remaining("anthropic") <= 600

    def test_nonpositive_block_uses_default(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path, default_block_seconds=45)
        guard.report_block("anthropic", -10, "rate_limit")
        remaining = guard.blocked_remaining("anthropic")
        assert 40 < remaining <= 45


class TestFailOpen:
    def test_corrupt_file_reads_as_unblocked(self, tmp_path: Path) -> None:
        path = tmp_path / "llm_rate_guard.json"
        path.write_text("{not valid json ::::")
        guard = _guard(tmp_path)
        assert guard.blocked_remaining("anthropic") == 0.0

    def test_report_over_corrupt_file_replaces_it(self, tmp_path: Path) -> None:
        path = tmp_path / "llm_rate_guard.json"
        path.write_text("garbage")
        guard = _guard(tmp_path)
        guard.report_block("anthropic", 60, "rate_limit")
        state = json.loads(path.read_text())
        assert "anthropic" in state

    def test_write_failure_is_swallowed(self, tmp_path: Path, monkeypatch) -> None:
        guard = _guard(tmp_path)

        def _boom(_state) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(guard, "_write_state", _boom)
        # Must not raise.
        guard.report_block("anthropic", 60, "rate_limit")


class TestEnabledToggle:
    def test_disabled_query_returns_zero_even_when_file_blocks(self, tmp_path: Path) -> None:
        path = tmp_path / "llm_rate_guard.json"
        path.write_text(json.dumps({"anthropic": {"blocked_until": time.time() + 300, "reason": "rate_limit"}}))
        guard = _guard(tmp_path, enabled=False)
        assert guard.blocked_remaining("anthropic") == 0.0

    def test_disabled_report_is_noop(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path, enabled=False)
        guard.report_block("anthropic", 60, "rate_limit")
        assert not (tmp_path / "llm_rate_guard.json").exists()


class TestConcurrency:
    def test_concurrent_writes_stay_valid_json(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path)
        families = [f"provider-{i}" for i in range(8)]

        def _writer(fam: str) -> None:
            for _ in range(25):
                guard.report_block(fam, 60, "rate_limit")

        threads = [threading.Thread(target=_writer, args=(fam,)) for fam in families]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # File must remain parseable (atomic replace, last-writer-wins).
        state = json.loads((tmp_path / "llm_rate_guard.json").read_text())
        assert isinstance(state, dict)
        # At least one family recorded; concurrent replaces may drop others.
        assert any(fam in state for fam in families)

    def test_flock_serializes_all_families_survive(self, tmp_path: Path) -> None:
        # With the sidecar-lock RMW, concurrent writes for distinct families
        # must not drop each other's entries.
        guard = _guard(tmp_path)
        families = [f"provider-{i}" for i in range(8)]

        def _writer(fam: str) -> None:
            for _ in range(30):
                guard.report_block(fam, 60, "rate_limit")

        threads = [threading.Thread(target=_writer, args=(fam,)) for fam in families]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        state = json.loads((tmp_path / "llm_rate_guard.json").read_text())
        assert set(families) <= set(state)


class TestLockFailOpen:
    def test_report_block_without_fcntl_still_writes(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(rate_guard, "fcntl", None)
        guard = _guard(tmp_path)
        guard.report_block("anthropic", 60, "rate_limit")
        assert guard.blocked_remaining("anthropic") > 0

    def test_flock_acquire_failure_falls_open(self, tmp_path: Path, monkeypatch) -> None:
        class _StubFcntl:
            LOCK_EX = 2
            LOCK_UN = 8

            @staticmethod
            def flock(fd, op):  # noqa: ANN001
                raise OSError("flock unavailable")

        monkeypatch.setattr(rate_guard, "fcntl", _StubFcntl)
        guard = _guard(tmp_path)
        # Must still record the block despite the lock failure.
        guard.report_block("anthropic", 60, "rate_limit")
        assert guard.blocked_remaining("anthropic") > 0


class TestReadPaths:
    def test_blocked_remaining_read_error_fails_open(self, tmp_path: Path, monkeypatch) -> None:
        guard = _guard(tmp_path)

        def _boom() -> dict:
            raise OSError("stat exploded")

        monkeypatch.setattr(guard, "_read_state_cached", _boom)
        assert guard.blocked_remaining("anthropic") == 0.0

    def test_report_block_read_before_write_error_still_writes(self, tmp_path: Path, monkeypatch) -> None:
        guard = _guard(tmp_path)

        def _boom() -> dict:
            raise ValueError("corrupt reader")

        monkeypatch.setattr(guard, "_read_state", _boom)
        guard.report_block("anthropic", 60, "rate_limit")
        # Reader raised, but the write started fresh and recorded the block.
        state = json.loads((tmp_path / "llm_rate_guard.json").read_text())
        assert "anthropic" in state

    def test_mtime_cache_avoids_reparse(self, tmp_path: Path, monkeypatch) -> None:
        guard = _guard(tmp_path)
        guard.report_block("anthropic", 300, "rate_limit")

        calls = {"n": 0}
        real_read = guard._read_state

        def _counting_read() -> dict:
            calls["n"] += 1
            return real_read()

        monkeypatch.setattr(guard, "_read_state", _counting_read)
        guard.blocked_remaining("anthropic")
        guard.blocked_remaining("anthropic")
        guard.blocked_remaining("anthropic")
        # Same mtime → parsed once, served from cache thereafter.
        assert calls["n"] == 1


class TestConfigResolution:
    def test_explicit_config_is_authoritative(self, tmp_path: Path) -> None:
        cfg = LlmRateGuardConfig(enabled=False)
        guard = LlmRateGuard(config=cfg, path=tmp_path / "g.json")
        assert guard.config is cfg

    def test_auto_config_resolves_each_ttl(self, tmp_path: Path, monkeypatch) -> None:
        resolved = LlmRateGuardConfig(default_block_seconds=99)
        monkeypatch.setattr(rate_guard, "_load_guard_config", lambda: resolved)
        guard = LlmRateGuard(path=tmp_path / "g.json")  # no explicit config
        assert guard.config.default_block_seconds == 99

    def test_load_guard_config_falls_back_on_error(self, monkeypatch) -> None:
        import core.config as config_mod

        monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        cfg = _load_guard_config()
        assert isinstance(cfg, LlmRateGuardConfig)
        assert cfg.enabled is True
