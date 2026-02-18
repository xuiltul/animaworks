# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""AnimaWorks AWS collector tool — ECS status, CloudWatch logs & metrics.

Provides a simplified, stateless interface to AWS ECS and CloudWatch.
All cluster/service/log-group parameters are passed per-call (no config.yaml).

Requires ``boto3``.  Install via::

    pip install animaworks[aws]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from core.tools._base import logger

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "ecs-status":  {"expected_seconds": 15, "background_eligible": False},
    "error-logs":  {"expected_seconds": 20, "background_eligible": False},
    "metrics":     {"expected_seconds": 15, "background_eligible": False},
}

# ---------------------------------------------------------------------------
# Guard boto3 import
# ---------------------------------------------------------------------------
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[assignment,misc]

_MISSING_MSG = "aws_collector requires 'boto3'. Install with: pip install animaworks[aws]"

DEFAULT_ERROR_PATTERNS: list[str] = [
    "Exception",
    "Error",
    "FATAL",
    "OOM",
    "OutOfMemory",
    "Timeout",
]


# ──────────────────────────────────────────────────────────────────────────────
# AWSCollector
# ──────────────────────────────────────────────────────────────────────────────

class AWSCollector:
    """Stateless AWS data collection for ECS services and CloudWatch."""

    def __init__(self, region: str | None = None) -> None:
        """Initialise AWS clients.

        Args:
            region: AWS region name.  Falls back to ``AWS_REGION`` env var,
                    then ``"ap-northeast-1"``.
        """
        if boto3 is None:
            raise ImportError(_MISSING_MSG)

        self.region = region or os.environ.get("AWS_REGION", "ap-northeast-1")
        self._ecs = boto3.client("ecs", region_name=self.region)
        self._logs = boto3.client("logs", region_name=self.region)
        self._cw = boto3.client("cloudwatch", region_name=self.region)
        logger.debug("AWSCollector initialised (region=%s)", self.region)

    # ── ECS ────────────────────────────────────────────────────────────────

    def get_ecs_status(self, cluster: str, service: str) -> dict[str, Any]:
        """Return current ECS service status.

        Args:
            cluster: ECS cluster name or ARN.
            service: ECS service name.

        Returns:
            Dict with ``runningCount``, ``desiredCount``, ``pendingCount``,
            ``status``, and ``deployments``.
        """
        if boto3 is None:
            raise ImportError(_MISSING_MSG)
        try:
            resp = self._ecs.describe_services(
                cluster=cluster, services=[service]
            )
            if not resp.get("services"):
                return {"error": f"Service '{service}' not found in cluster '{cluster}'"}

            svc = resp["services"][0]
            deployments = []
            for dep in svc.get("deployments", []):
                deployments.append({
                    "status": dep.get("status"),
                    "runningCount": dep.get("runningCount", 0),
                    "desiredCount": dep.get("desiredCount", 0),
                    "taskDefinition": dep.get("taskDefinition", ""),
                })

            return {
                "runningCount": svc.get("runningCount", 0),
                "desiredCount": svc.get("desiredCount", 0),
                "pendingCount": svc.get("pendingCount", 0),
                "status": svc.get("status", "UNKNOWN"),
                "deployments": deployments,
            }
        except ClientError as exc:
            logger.error("get_ecs_status failed: %s", exc)
            return {"error": str(exc)}

    def get_ecs_events(
        self, cluster: str, service: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return recent ECS service events (task restarts, health-check failures, etc.).

        Args:
            cluster: ECS cluster name or ARN.
            service: ECS service name.
            limit:   Maximum number of events to return.

        Returns:
            List of dicts with ``createdAt`` (ISO-8601) and ``message``.
        """
        if boto3 is None:
            raise ImportError(_MISSING_MSG)
        try:
            resp = self._ecs.describe_services(
                cluster=cluster, services=[service]
            )
            if not resp.get("services"):
                return []

            events = resp["services"][0].get("events", [])[:limit]
            return [
                {
                    "createdAt": (
                        ev["createdAt"].isoformat()
                        if isinstance(ev.get("createdAt"), datetime)
                        else str(ev.get("createdAt", ""))
                    ),
                    "message": ev.get("message", ""),
                }
                for ev in events
            ]
        except ClientError as exc:
            logger.error("get_ecs_events failed: %s", exc)
            return []

    # ── CloudWatch Logs ────────────────────────────────────────────────────

    def get_error_logs(
        self,
        log_group: str,
        hours: int = 1,
        patterns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search CloudWatch Logs for error entries.

        Args:
            log_group: CloudWatch log group name.
            hours:     How far back to search (in hours).
            patterns:  Keywords to match.  Defaults to common error patterns.

        Returns:
            List of dicts with ``timestamp`` (epoch-ms), ``datetime`` (ISO-8601),
            and ``message``.
        """
        if boto3 is None:
            raise ImportError(_MISSING_MSG)

        kw = patterns if patterns is not None else DEFAULT_ERROR_PATTERNS
        filter_pattern = " ".join(f"?{p}" for p in kw)

        start_ms = int(
            (datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() * 1000
        )

        try:
            resp = self._logs.filter_log_events(
                logGroupName=log_group,
                startTime=start_ms,
                filterPattern=filter_pattern,
                limit=100,
            )
            events = resp.get("events", [])
            return [
                {
                    "timestamp": ev.get("timestamp", 0),
                    "datetime": (
                        datetime.fromtimestamp(
                            ev["timestamp"] / 1000, tz=timezone.utc
                        ).isoformat()
                        if ev.get("timestamp")
                        else None
                    ),
                    "message": ev.get("message", ""),
                }
                for ev in events
            ]
        except ClientError as exc:
            logger.error("get_error_logs failed: %s", exc)
            return []

    # ── CloudWatch Metrics ─────────────────────────────────────────────────

    def get_metrics(
        self,
        cluster: str,
        service: str,
        metric: str = "CPUUtilization",
        hours: int = 1,
    ) -> dict[str, Any]:
        """Fetch CloudWatch metrics for an ECS service.

        Args:
            cluster: ECS cluster name.
            service: ECS service name.
            metric:  Metric name (``CPUUtilization``, ``MemoryUtilization``, etc.).
            hours:   Time window in hours.

        Returns:
            Dict with ``datapoints`` list and ``summary`` stats.
        """
        if boto3 is None:
            raise ImportError(_MISSING_MSG)

        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)

        try:
            resp = self._cw.get_metric_statistics(
                Namespace="ECS/ContainerInsights",
                MetricName=metric,
                Dimensions=[
                    {"Name": "ClusterName", "Value": cluster},
                    {"Name": "ServiceName", "Value": service},
                ],
                StartTime=start,
                EndTime=end,
                Period=300,
                Statistics=["Average", "Maximum"],
            )

            raw = sorted(
                resp.get("Datapoints", []),
                key=lambda d: d.get("Timestamp", datetime.min),
            )

            datapoints = [
                {
                    "timestamp": (
                        dp["Timestamp"].isoformat()
                        if isinstance(dp.get("Timestamp"), datetime)
                        else str(dp.get("Timestamp", ""))
                    ),
                    "average": dp.get("Average"),
                    "maximum": dp.get("Maximum"),
                    "unit": dp.get("Unit", ""),
                }
                for dp in raw
            ]

            averages = [dp["Average"] for dp in raw if dp.get("Average") is not None]
            maximums = [dp["Maximum"] for dp in raw if dp.get("Maximum") is not None]

            summary: dict[str, Any] = {
                "metric": metric,
                "datapointCount": len(raw),
            }
            if averages:
                summary["average"] = round(sum(averages) / len(averages), 2)
            if maximums:
                summary["maximum"] = round(max(maximums), 2)

            return {"datapoints": datapoints, "summary": summary}

        except ClientError as exc:
            logger.error("get_metrics failed: %s", exc)
            return {"datapoints": [], "summary": {"error": str(exc)}}


# ──────────────────────────────────────────────────────────────────────────────
# Tool schemas (Anthropic tool_use format)
# ──────────────────────────────────────────────────────────────────────────────

def get_tool_schemas() -> list[dict[str, Any]]:
    """Return Anthropic-compatible tool schemas for the AWS collector."""
    return [
        {
            "name": "aws_ecs_status",
            "description": (
                "Get current status of an AWS ECS service: running/desired task "
                "counts, deployment state."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "cluster": {
                        "type": "string",
                        "description": "ECS cluster name or ARN.",
                    },
                    "service": {
                        "type": "string",
                        "description": "ECS service name.",
                    },
                },
                "required": ["cluster", "service"],
            },
        },
        {
            "name": "aws_error_logs",
            "description": (
                "Search CloudWatch Logs for error-level entries "
                "(Exception, FATAL, OOM, Timeout, etc.)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "log_group": {
                        "type": "string",
                        "description": "CloudWatch log group name.",
                    },
                    "hours": {
                        "type": "integer",
                        "description": "How many hours back to search. Default 1.",
                        "default": 1,
                    },
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Custom filter keywords (default: common error patterns).",
                    },
                },
                "required": ["log_group"],
            },
        },
        {
            "name": "aws_metrics",
            "description": (
                "Fetch CloudWatch metrics (CPU, Memory, etc.) for an ECS service."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "cluster": {
                        "type": "string",
                        "description": "ECS cluster name.",
                    },
                    "service": {
                        "type": "string",
                        "description": "ECS service name.",
                    },
                    "metric": {
                        "type": "string",
                        "description": "Metric name (default: CPUUtilization).",
                        "default": "CPUUtilization",
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Time window in hours. Default 1.",
                        "default": 1,
                    },
                },
                "required": ["cluster", "service"],
            },
        },
    ]


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────


def get_cli_guide() -> str:
    """Return CLI usage guide for AWS monitoring tools."""
    return """\
### AWS モニタリング
```bash
animaworks-tool aws_collector ecs-status --cluster <クラスタ> --service <サービス> -j
animaworks-tool aws_collector error-logs --log-group <ロググループ> --hours 2 -j
animaworks-tool aws_collector metrics --cluster <クラスタ> --service <サービス> --metric CPUUtilization -j
```"""


def cli_main(argv: list[str] | None = None) -> None:
    """Standalone CLI for AWS data collection.

    Sub-commands::

        ecs-status  --cluster NAME --service NAME
        error-logs  --log-group NAME [--hours N] [--patterns P1 P2 ...]
        metrics     --cluster NAME --service NAME [--metric M] [--hours N]
    """
    parser = argparse.ArgumentParser(
        prog="animaworks-aws",
        description="AnimaWorks AWS collector CLI",
    )
    parser.add_argument(
        "--region", default=None, help="AWS region (default: AWS_REGION env or ap-northeast-1)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ecs-status
    p_ecs = sub.add_parser("ecs-status", help="Show ECS service status")
    p_ecs.add_argument("--cluster", required=True)
    p_ecs.add_argument("--service", required=True)

    # error-logs
    p_logs = sub.add_parser("error-logs", help="Search CloudWatch error logs")
    p_logs.add_argument("--log-group", required=True)
    p_logs.add_argument("--hours", type=int, default=1)
    p_logs.add_argument("--patterns", nargs="*", default=None)

    # metrics
    p_met = sub.add_parser("metrics", help="Fetch CloudWatch metrics")
    p_met.add_argument("--cluster", required=True)
    p_met.add_argument("--service", required=True)
    p_met.add_argument("--metric", default="CPUUtilization")
    p_met.add_argument("--hours", type=int, default=1)

    args = parser.parse_args(argv)
    collector = AWSCollector(region=args.region)

    if args.command == "ecs-status":
        result = collector.get_ecs_status(args.cluster, args.service)
    elif args.command == "error-logs":
        result = collector.get_error_logs(
            args.log_group, hours=args.hours, patterns=args.patterns
        )
    elif args.command == "metrics":
        result = collector.get_metrics(
            args.cluster, args.service, metric=args.metric, hours=args.hours
        )
    else:
        parser.print_help()
        sys.exit(1)

    json.dump(result, sys.stdout, indent=2, ensure_ascii=False, default=str)
    print()  # trailing newline


# ── Dispatch ──────────────────────────────────────────


def dispatch(tool_name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call to the appropriate handler."""
    if tool_name == "aws_ecs_status":
        collector = AWSCollector(region=args.get("region"))
        return collector.get_ecs_status(args["cluster"], args["service"])
    if tool_name == "aws_error_logs":
        collector = AWSCollector(region=args.get("region"))
        return collector.get_error_logs(
            log_group=args["log_group"],
            hours=args.get("hours", 1),
            patterns=args.get("patterns"),
        )
    if tool_name == "aws_metrics":
        collector = AWSCollector(region=args.get("region"))
        return collector.get_metrics(
            cluster=args["cluster"],
            service=args["service"],
            metric=args.get("metric", "CPUUtilization"),
            hours=args.get("hours", 1),
        )
    raise ValueError(f"Unknown tool: {tool_name}")


if __name__ == "__main__":
    cli_main()