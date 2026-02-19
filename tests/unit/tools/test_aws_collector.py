"""Tests for core/tools/aws_collector.py — AWS ECS/CloudWatch integration."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from botocore.exceptions import ClientError as BotoClientError

from core.tools.aws_collector import (
    AWSCollector,
    DEFAULT_ERROR_PATTERNS,
    get_tool_schemas,
)


def _make_client_error(message: str = "Access Denied") -> BotoClientError:
    """Create a botocore ClientError for testing."""
    return BotoClientError(
        error_response={"Error": {"Code": "AccessDeniedException", "Message": message}},
        operation_name="TestOperation",
    )


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def mock_boto3():
    """Provide mocked boto3 clients."""
    mock_ecs = MagicMock()
    mock_logs = MagicMock()
    mock_cw = MagicMock()

    with patch("core.tools.aws_collector.boto3") as mock_b3:
        mock_b3.client.side_effect = lambda svc, **kwargs: {
            "ecs": mock_ecs,
            "logs": mock_logs,
            "cloudwatch": mock_cw,
        }.get(svc, MagicMock())
        yield {"ecs": mock_ecs, "logs": mock_logs, "cloudwatch": mock_cw, "boto3": mock_b3}


# ── AWSCollector init ─────────────────────────────────────────────


class TestAWSCollectorInit:
    def test_init_with_region(self, mock_boto3):
        collector = AWSCollector(region="us-west-2")
        assert collector.region == "us-west-2"

    def test_init_default_region(self, mock_boto3, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        collector = AWSCollector()
        assert collector.region == "ap-northeast-1"

    def test_init_region_from_env(self, mock_boto3, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        collector = AWSCollector()
        assert collector.region == "eu-west-1"

    def test_init_without_boto3(self, monkeypatch: pytest.MonkeyPatch):
        with patch("core.tools.aws_collector.boto3", None):
            with pytest.raises(ImportError, match="boto3"):
                AWSCollector()


# ── get_ecs_status ────────────────────────────────────────────────


class TestGetEcsStatus:
    def test_success(self, mock_boto3):
        mock_boto3["ecs"].describe_services.return_value = {
            "services": [
                {
                    "runningCount": 2,
                    "desiredCount": 2,
                    "pendingCount": 0,
                    "status": "ACTIVE",
                    "deployments": [
                        {
                            "status": "PRIMARY",
                            "runningCount": 2,
                            "desiredCount": 2,
                            "taskDefinition": "arn:aws:ecs:us-east-1:123:task-def/my-task:1",
                        }
                    ],
                }
            ]
        }
        collector = AWSCollector()
        result = collector.get_ecs_status("my-cluster", "my-service")
        assert result["runningCount"] == 2
        assert result["status"] == "ACTIVE"
        assert len(result["deployments"]) == 1

    def test_service_not_found(self, mock_boto3):
        mock_boto3["ecs"].describe_services.return_value = {"services": []}
        collector = AWSCollector()
        result = collector.get_ecs_status("cluster", "missing-service")
        assert "error" in result

    def test_client_error(self, mock_boto3):
        mock_boto3["ecs"].describe_services.side_effect = _make_client_error("Access Denied")
        collector = AWSCollector()
        result = collector.get_ecs_status("cluster", "service")
        assert "error" in result


# ── get_ecs_events ────────────────────────────────────────────────


class TestGetEcsEvents:
    def test_success(self, mock_boto3):
        mock_boto3["ecs"].describe_services.return_value = {
            "services": [
                {
                    "events": [
                        {
                            "createdAt": datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
                            "message": "service reached steady state",
                        },
                        {
                            "createdAt": "2026-01-15T11:00:00Z",
                            "message": "task started",
                        },
                    ],
                }
            ]
        }
        collector = AWSCollector()
        events = collector.get_ecs_events("cluster", "service", limit=10)
        assert len(events) == 2
        assert "steady state" in events[0]["message"]

    def test_service_not_found(self, mock_boto3):
        mock_boto3["ecs"].describe_services.return_value = {"services": []}
        collector = AWSCollector()
        events = collector.get_ecs_events("cluster", "missing")
        assert events == []

    def test_error(self, mock_boto3):
        mock_boto3["ecs"].describe_services.side_effect = _make_client_error("err")
        collector = AWSCollector()
        events = collector.get_ecs_events("cluster", "service")
        assert events == []


# ── get_error_logs ────────────────────────────────────────────────


class TestGetErrorLogs:
    def test_success(self, mock_boto3):
        mock_boto3["logs"].filter_log_events.return_value = {
            "events": [
                {"timestamp": 1707000000000, "message": "java.lang.Exception: oops"},
                {"timestamp": 1707000001000, "message": "FATAL: out of memory"},
            ]
        }
        collector = AWSCollector()
        logs = collector.get_error_logs("/ecs/my-service", hours=2)
        assert len(logs) == 2
        assert logs[0]["datetime"] is not None
        assert "Exception" in logs[0]["message"]

    def test_custom_patterns(self, mock_boto3):
        mock_boto3["logs"].filter_log_events.return_value = {"events": []}
        collector = AWSCollector()
        collector.get_error_logs("/ecs/svc", patterns=["CustomError", "Panic"])
        call_kwargs = mock_boto3["logs"].filter_log_events.call_args.kwargs
        assert "CustomError" in call_kwargs["filterPattern"]
        assert "Panic" in call_kwargs["filterPattern"]

    def test_default_patterns(self, mock_boto3):
        mock_boto3["logs"].filter_log_events.return_value = {"events": []}
        collector = AWSCollector()
        collector.get_error_logs("/ecs/svc")
        call_kwargs = mock_boto3["logs"].filter_log_events.call_args.kwargs
        for p in DEFAULT_ERROR_PATTERNS:
            assert f"?{p}" in call_kwargs["filterPattern"]

    def test_error(self, mock_boto3):
        mock_boto3["logs"].filter_log_events.side_effect = _make_client_error("err")
        collector = AWSCollector()
        logs = collector.get_error_logs("/ecs/svc")
        assert logs == []

    def test_missing_timestamp(self, mock_boto3):
        mock_boto3["logs"].filter_log_events.return_value = {
            "events": [{"timestamp": 0, "message": "test"}]
        }
        collector = AWSCollector()
        logs = collector.get_error_logs("/ecs/svc")
        assert logs[0]["datetime"] is None


# ── get_metrics ───────────────────────────────────────────────────


class TestGetMetrics:
    def test_success(self, mock_boto3):
        mock_boto3["cloudwatch"].get_metric_statistics.return_value = {
            "Datapoints": [
                {
                    "Timestamp": datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
                    "Average": 25.5,
                    "Maximum": 40.0,
                    "Unit": "Percent",
                },
                {
                    "Timestamp": datetime(2026, 1, 15, 12, 5, tzinfo=timezone.utc),
                    "Average": 30.0,
                    "Maximum": 45.0,
                    "Unit": "Percent",
                },
            ]
        }
        collector = AWSCollector()
        result = collector.get_metrics("cluster", "service")
        assert len(result["datapoints"]) == 2
        assert result["summary"]["datapointCount"] == 2
        assert result["summary"]["average"] == 27.75
        assert result["summary"]["maximum"] == 45.0

    def test_no_datapoints(self, mock_boto3):
        mock_boto3["cloudwatch"].get_metric_statistics.return_value = {"Datapoints": []}
        collector = AWSCollector()
        result = collector.get_metrics("cluster", "service")
        assert result["datapoints"] == []
        assert result["summary"]["datapointCount"] == 0

    def test_error(self, mock_boto3):
        mock_boto3["cloudwatch"].get_metric_statistics.side_effect = _make_client_error("err")
        collector = AWSCollector()
        result = collector.get_metrics("cluster", "service")
        assert result["datapoints"] == []
        assert "error" in result["summary"]

    def test_custom_metric(self, mock_boto3):
        mock_boto3["cloudwatch"].get_metric_statistics.return_value = {"Datapoints": []}
        collector = AWSCollector()
        collector.get_metrics("c", "s", metric="MemoryUtilization", hours=6)
        call_kwargs = mock_boto3["cloudwatch"].get_metric_statistics.call_args.kwargs
        assert call_kwargs["MetricName"] == "MemoryUtilization"


# ── get_tool_schemas ──────────────────────────────────────────────


class TestGetToolSchemas:
    def test_returns_schemas(self):
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 3
        names = {s["name"] for s in schemas}
        assert names == {"aws_ecs_status", "aws_error_logs", "aws_metrics"}

    def test_ecs_status_requires_cluster_and_service(self):
        schemas = get_tool_schemas()
        ecs = [s for s in schemas if s["name"] == "aws_ecs_status"][0]
        assert set(ecs["input_schema"]["required"]) == {"cluster", "service"}

    def test_error_logs_requires_log_group(self):
        schemas = get_tool_schemas()
        logs = [s for s in schemas if s["name"] == "aws_error_logs"][0]
        assert "log_group" in logs["input_schema"]["required"]
