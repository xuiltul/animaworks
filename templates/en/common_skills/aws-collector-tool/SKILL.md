---
name: aws-collector-tool
description: >-
  AWS infrastructure monitoring tool. ECS status, CloudWatch error logs, and metrics.
  "AWS" "ECS" "CloudWatch" "infrastructure" "metrics" "logs"
tags: [infrastructure, aws, monitoring, external]
---

# AWS Collector Tool

External tool for collecting AWS ECS status, CloudWatch logs, and metrics.

## Invocation via Bash

Use **Bash** with `animaworks-tool aws_collector <subcommand> [args]`. See Actions below for syntax.

## Actions

### ecs_status — ECS service status
```json
{"tool_name": "aws_collector", "action": "ecs_status", "args": {"cluster": "cluster-name", "service": "service-name (optional)"}}
```

### error_logs — CloudWatch error logs
```json
{"tool_name": "aws_collector", "action": "error_logs", "args": {"log_group": "log-group-name", "hours": 1, "patterns": "ERROR,Exception"}}
```

### metrics — CloudWatch metrics
```json
{"tool_name": "aws_collector", "action": "metrics", "args": {"cluster": "cluster-name", "service": "service-name", "metric": "CPUUtilization", "hours": 1}}
```

## CLI Usage (S/C/D/G-mode)

```bash
animaworks-tool aws_collector ecs-status [--cluster NAME] [--service NAME]
animaworks-tool aws_collector error-logs --log-group NAME [--hours 1] [--patterns "ERROR"]
animaworks-tool aws_collector metrics --cluster NAME --service NAME [--metric CPUUtilization]
```

## Notes

- AWS credentials must be configured (environment variables or credentials file)
- Use --region to specify AWS region
