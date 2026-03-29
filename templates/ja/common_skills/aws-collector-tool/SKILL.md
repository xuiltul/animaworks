---
name: aws-collector-tool
description: >-
  AWSインフラ情報収集ツール。ECS状態・CloudWatchログ・メトリクスを取得する。
  Use when: ECS稼働確認、CloudWatchでエラーログ調査、メトリクス取得、AWSリソース監視が必要なとき。
tags: [infrastructure, aws, monitoring, external]
---

# AWS Collector ツール

AWS ECSステータス・CloudWatchログ・メトリクスを収集する外部ツール。

## 呼び出し方法

**Bash**: `animaworks-tool aws_collector <サブコマンド> [引数]` で実行

## アクション一覧

### ecs_status — ECSサービス状態確認
```bash
animaworks-tool aws_collector ecs-status [--cluster NAME] [--service NAME]
```

### error_logs — エラーログ取得
```bash
animaworks-tool aws_collector error-logs --log-group NAME [--hours 1] [--patterns "ERROR"]
```

### metrics — メトリクス取得
```bash
animaworks-tool aws_collector metrics --cluster NAME --service NAME [--metric CPUUtilization]
```

## CLI使用法

```bash
animaworks-tool aws_collector ecs-status [--cluster NAME] [--service NAME]
animaworks-tool aws_collector error-logs --log-group NAME [--hours 1] [--patterns "ERROR"]
animaworks-tool aws_collector metrics --cluster NAME --service NAME [--metric CPUUtilization]
```

## 注意事項

- AWS認証情報（環境変数またはcredentials）の設定が必要
- --region でリージョン指定可能
