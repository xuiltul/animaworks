---
name: aws-collector-tool
description: >-
  AWS基础设施信息收集工具。获取ECS状态、CloudWatch日志、指标。
  Use when: 需要确认ECS运行状态、在CloudWatch中调查错误日志、获取指标、监控AWS资源时。
tags: [infrastructure, aws, monitoring, external]
---


# AWS Collector 工具

收集AWS ECS状态、CloudWatch日志、指标的外部工具。

## 调用方法

**Bash**: 通过 `animaworks-tool aws_collector <サブコマンド> [引数]` 执行

## 操作列表

### ecs_status — ECS服务状态确认
```bash
animaworks-tool aws_collector ecs-status [--cluster NAME] [--service NAME]
```

### error_logs — 获取错误日志
```bash
animaworks-tool aws_collector error-logs --log-group NAME [--hours 1] [--patterns "ERROR"]
```

### metrics — 获取指标
```bash
animaworks-tool aws_collector metrics --cluster NAME --service NAME [--metric CPUUtilization]
```

## CLI使用方法

```bash
animaworks-tool aws_collector ecs-status [--cluster NAME] [--service NAME]
animaworks-tool aws_collector error-logs --log-group NAME [--hours 1] [--patterns "ERROR"]
animaworks-tool aws_collector metrics --cluster NAME --service NAME [--metric CPUUtilization]
```

## 注意事项

- 需要配置AWS认证信息（环境变量或credentials）
- 可通过 --region 指定区域
