from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for activity report API routes."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.audit import AnimaAuditEntry, OrgAuditReport
from server.routes.activity_report import create_activity_report_router


@pytest.fixture()
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(create_activity_report_router(), prefix="/api")
    return app


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _sample_report(date: str = "2026-03-07") -> OrgAuditReport:
    return OrgAuditReport(
        date=date,
        animas=[
            AnimaAuditEntry(
                name="alice",
                enabled=True,
                model="claude-sonnet-4-6",
                supervisor=None,
                role="engineer",
                total_entries=100,
                type_counts={"heartbeat_start": 10},
                messages_sent=5,
                messages_received=3,
                errors=1,
                tasks_total=10,
                tasks_pending=2,
                tasks_done=7,
                peers_sent={"bob": 3},
                peers_received={"carol": 2},
                first_activity="2026-03-07T09:00:00+09:00",
                last_activity="2026-03-07T18:00:00+09:00",
            ),
        ],
        total_entries=100,
        total_messages=8,
        total_errors=1,
        total_tasks_done=7,
        active_anima_count=1,
        disabled_anima_count=0,
    )


# ── GET /models ───────────────────────────────────────────────


class TestModelsEndpoint:
    def test_returns_models(self, client: TestClient):
        with (
            patch("server.routes.activity_report._resolve_model", return_value="anthropic/claude-sonnet-4-6"),
            patch(
                "server.routes.activity_report._available_models",
                return_value=[
                    {"id": "anthropic/claude-sonnet-4-6", "label": "claude-sonnet-4-6"},
                ],
            ),
        ):
            res = client.get("/api/activity-report/models")

        assert res.status_code == 200
        data = res.json()
        assert data["default_model"] == "anthropic/claude-sonnet-4-6"
        assert len(data["available_models"]) == 1


# ── POST /generate ────────────────────────────────────────────


class TestGenerateEndpoint:
    def test_invalid_date_format(self, client: TestClient):
        res = client.post(
            "/api/activity-report/generate",
            json={"date": "not-a-date"},
        )
        assert res.status_code == 422  # Pydantic validation

    def test_future_date_rejected(self, client: TestClient):
        with (
            patch("server.routes.activity_report._resolve_model", return_value="test"),
            patch("server.routes.activity_report.now_jst") as mock_now,
        ):
            jst = timezone(timedelta(hours=9))
            mock_now.return_value = datetime(2026, 3, 7, 12, 0, tzinfo=jst)
            res = client.post(
                "/api/activity-report/generate",
                json={"date": "2030-12-31"},
            )

        assert res.status_code == 400
        assert "error" in res.json()

    def test_generates_report(self, client: TestClient, tmp_path: Path):
        report = _sample_report()

        with (
            patch("server.routes.activity_report._resolve_model", return_value="test-model"),
            patch("server.routes.activity_report._read_cache", return_value=None),
            patch("server.routes.activity_report._write_cache"),
            patch("core.audit.collect_org_audit", new_callable=AsyncMock, return_value=report),
            patch("core.audit.generate_org_timeline", return_value="[09:00] alice HB\n  checking"),
            patch(
                "server.routes.activity_report._generate_narrative",
                new_callable=AsyncMock,
                return_value="# Report\nAll good",
            ),
            patch("server.routes.activity_report.now_jst") as mock_now,
        ):
            jst = timezone(timedelta(hours=9))
            mock_now.return_value = datetime(2026, 3, 7, 20, 0, tzinfo=jst)

            res = client.post(
                "/api/activity-report/generate",
                json={"date": "2026-03-07"},
            )

        assert res.status_code == 200
        data = res.json()
        assert data["date"] == "2026-03-07"
        assert data["structured"]["total_entries"] == 100
        assert data["narrative_md"] == "# Report\nAll good"
        assert data["cached"] is False

    def test_returns_cached_report(self, client: TestClient):
        cached_data = {"date": "2026-03-07", "structured": {}, "narrative_md": "cached", "model_used": "x"}

        with (
            patch("server.routes.activity_report._resolve_model", return_value="x"),
            patch("server.routes.activity_report._read_cache", return_value=cached_data),
            patch("server.routes.activity_report.now_jst") as mock_now,
        ):
            jst = timezone(timedelta(hours=9))
            mock_now.return_value = datetime(2026, 3, 7, 20, 0, tzinfo=jst)

            res = client.post(
                "/api/activity-report/generate",
                json={"date": "2026-03-07"},
            )

        assert res.status_code == 200
        assert res.json()["cached"] is True

    def test_force_regenerate_bypasses_cache(self, client: TestClient):
        report = _sample_report()

        with (
            patch("server.routes.activity_report._resolve_model", return_value="test"),
            patch("server.routes.activity_report._read_cache") as mock_cache,
            patch("server.routes.activity_report._write_cache"),
            patch("core.audit.collect_org_audit", new_callable=AsyncMock, return_value=report),
            patch("core.audit.generate_org_timeline", return_value=""),
            patch("server.routes.activity_report._generate_narrative", new_callable=AsyncMock, return_value=None),
            patch("server.routes.activity_report.now_jst") as mock_now,
        ):
            jst = timezone(timedelta(hours=9))
            mock_now.return_value = datetime(2026, 3, 7, 20, 0, tzinfo=jst)

            res = client.post(
                "/api/activity-report/generate",
                json={"date": "2026-03-07", "force_regenerate": True},
            )

        mock_cache.assert_not_called()
        assert res.status_code == 200

    def test_narrative_null_when_no_entries(self, client: TestClient):
        empty_report = OrgAuditReport(date="2026-03-07", animas=[], total_entries=0)

        with (
            patch("server.routes.activity_report._resolve_model", return_value="test"),
            patch("server.routes.activity_report._read_cache", return_value=None),
            patch("server.routes.activity_report._write_cache"),
            patch("core.audit.collect_org_audit", new_callable=AsyncMock, return_value=empty_report),
            patch("core.audit.generate_org_timeline", return_value=""),
            patch("server.routes.activity_report.now_jst") as mock_now,
        ):
            jst = timezone(timedelta(hours=9))
            mock_now.return_value = datetime(2026, 3, 7, 20, 0, tzinfo=jst)

            res = client.post(
                "/api/activity-report/generate",
                json={"date": "2026-03-07"},
            )

        assert res.status_code == 200
        assert res.json()["narrative_md"] is None


# ── GET /{report_date} ───────────────────────────────────────


class TestCachedReportEndpoint:
    def test_invalid_date(self, client: TestClient):
        res = client.get("/api/activity-report/bad-date")
        assert res.status_code == 400

    def test_not_found(self, client: TestClient, tmp_path: Path):
        with patch("server.routes.activity_report._cache_dir", return_value=tmp_path):
            res = client.get("/api/activity-report/2026-03-07")
        assert res.status_code == 404

    def test_returns_cached(self, client: TestClient, tmp_path: Path):
        cache_file = tmp_path / "2026-03-07_abcd1234.json"
        cache_data = {"date": "2026-03-07", "structured": {}, "narrative_md": "test"}
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        with patch("server.routes.activity_report._cache_dir", return_value=tmp_path):
            res = client.get("/api/activity-report/2026-03-07")

        assert res.status_code == 200
        data = res.json()
        assert data["date"] == "2026-03-07"
        assert data["cached"] is True
