---
name: aws-collector-tool
description: >-
  AWS 인프라 정보 수집 도구. ECS 상태·CloudWatch 로그·메트릭스를 가져온다.
  Use when: ECS 상태 확인, CloudWatch 오류 로그 조사, 메트릭스 조회, AWS 리소스 모니터링이 필요할 때.
tags: [infrastructure, aws, monitoring, external]
---

# AWS Collector 도구

AWS ECS 상태, CloudWatch 로그, 메트릭스를 수집하는 외부 도구입니다.

## 호출 방법

**Bash**: `animaworks-tool aws_collector <서브커맨드> [인수]`로 실행합니다.

## 액션 목록

### ecs_status — ECS 서비스 상태 확인
```bash
animaworks-tool aws_collector ecs-status [--cluster NAME] [--service NAME]
```

### error_logs — 에러 로그 취득
```bash
animaworks-tool aws_collector error-logs --log-group NAME [--hours 1] [--patterns "ERROR"]
```

### metrics — 메트릭스 취득
```bash
animaworks-tool aws_collector metrics --cluster NAME --service NAME [--metric CPUUtilization]
```

## CLI 사용법

```bash
animaworks-tool aws_collector ecs-status [--cluster NAME] [--service NAME]
animaworks-tool aws_collector error-logs --log-group NAME [--hours 1] [--patterns "ERROR"]
animaworks-tool aws_collector metrics --cluster NAME --service NAME [--metric CPUUtilization]
```

## 주의사항

- AWS 인증 정보(환경변수 또는 credentials 파일)가 설정되어 있어야 합니다
- --region으로 리전 지정 가능
