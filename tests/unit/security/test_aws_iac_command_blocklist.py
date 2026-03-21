"""Tests for AWS CLI / IaC destructive command blocklist patterns.

Covers:
- 16 deny patterns (Tier 1-7: credential escalation, RCE, destructive ops, IaC)
- 18 read-only commands NOT blocked (false-positive prevention)
- animaworks-tool aws_collector passthrough
- rin incident commands (2026-03-20/21)
- DataTalks.Club terraform destroy scenario
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re

import pytest

from core.config.global_permissions import _compile_patterns
from core.config.schemas import GlobalPermissionsConfig
from core.paths import TEMPLATES_DIR

# ── helpers ───────────────────────────────────────────────────


@pytest.fixture()
def deny_patterns() -> list[tuple[re.Pattern[str], str]]:
    """Compiled deny patterns from the default template."""
    src = TEMPLATES_DIR / "_shared" / "config_defaults" / "permissions.global.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    config = GlobalPermissionsConfig.model_validate(data)
    return _compile_patterns(config.commands.deny)


def _blocked(patterns: list[tuple[re.Pattern[str], str]], cmd: str) -> bool:
    return any(p.search(cmd) for p, _ in patterns)


# ── Tier 1: AWS credential escalation ────────────────────────


class TestTier1CredentialEscalation:
    @pytest.mark.parametrize(
        "cmd",
        [
            "aws sso login --profile future_sync_stg",
            "aws sso login",
        ],
    )
    def test_sso_login_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd), f"aws sso login not blocked: {cmd!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            "aws sts assume-role --role-arn arn:aws:iam::123:role/Admin",
            "aws sts assume-role --role-arn arn:aws:iam::456:role/deploy",
        ],
    )
    def test_sts_assume_role_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd), f"aws sts assume-role not blocked: {cmd!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            "aws configure set aws_access_key_id AKIA...",
            "aws configure import --csv file://creds.csv",
        ],
    )
    def test_configure_mutation_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd), f"aws configure mutation not blocked: {cmd!r}"


# ── Tier 2: AWS remote code execution ────────────────────────


class TestTier2RemoteCodeExecution:
    @pytest.mark.parametrize(
        "cmd",
        [
            "aws ecs execute-command --cluster stg --task abc --command '/bin/bash'",
            "aws ecs execute-command --cluster prod --task def --container app --command 'php artisan migrate:fresh'",
        ],
    )
    def test_ecs_execute_command_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd), f"ecs execute-command not blocked: {cmd!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            "aws ssm send-command --instance-ids i-1234 --document-name AWS-RunShellScript",
            "aws ssm send-command --targets Key=tag:Env,Values=prod --document-name cleanup",
        ],
    )
    def test_ssm_send_command_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd), f"ssm send-command not blocked: {cmd!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            "aws lambda invoke --function-name my-func output.json",
        ],
    )
    def test_lambda_invoke_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd), f"lambda invoke not blocked: {cmd!r}"


# ── Tier 3: AWS destructive generic ──────────────────────────


class TestTier3DestructiveGeneric:
    @pytest.mark.parametrize(
        "cmd",
        [
            "aws ec2 terminate-instances --instance-ids i-1234",
            "aws rds delete-db-instance --db-instance-identifier mydb --skip-final-snapshot",
            "aws ecs delete-service --cluster prod --service web --force",
            "aws ecs stop-task --cluster prod --task abc123",
            "aws ecs delete-cluster --cluster staging",
            "aws ecs deregister-task-definition --task-definition web:42",
            "aws elasticache delete-cache-cluster --cache-cluster-id my-cache",
        ],
    )
    def test_destructive_operations_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd), f"destructive op not blocked: {cmd!r}"


# ── Tier 4: AWS ECS mutating ─────────────────────────────────


class TestTier4ECSMutating:
    @pytest.mark.parametrize(
        "cmd",
        [
            "aws ecs run-task --cluster stg --task-definition prewarm:3",
            "aws ecs run-task --cluster stg --overrides '{\"containerOverrides\":[...]}'",
            "aws ecs update-service --cluster prod --service web --desired-count 0",
            "aws ecs update-service --cluster staging --service api --force-new-deployment",
        ],
    )
    def test_ecs_mutating_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd), f"ecs mutating not blocked: {cmd!r}"


# ── Tier 5: AWS S3 destructive ───────────────────────────────


class TestTier5S3Destructive:
    @pytest.mark.parametrize(
        "cmd",
        [
            "aws s3 rm s3://my-bucket/data/ --recursive",
            "aws s3 rb s3://my-bucket --force",
            "aws s3 mv s3://my-bucket/important/ s3://trash-bucket/",
        ],
    )
    def test_s3_destructive_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd), f"S3 destructive not blocked: {cmd!r}"


# ── Tier 6: AWS CloudFormation / IAM ─────────────────────────


class TestTier6CloudFormationIAM:
    @pytest.mark.parametrize(
        "cmd",
        [
            "aws cloudformation delete-stack --stack-name prod-infra",
            "aws cloudformation update-stack --stack-name prod-infra --template-body file://t.yaml",
        ],
    )
    def test_cloudformation_mutating_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd), f"CloudFormation op not blocked: {cmd!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            "aws iam create-user --user-name hacker",
            "aws iam delete-role --role-name admin",
            "aws iam put-role-policy --role-name admin --policy-name FullAccess --policy-document file://p.json",
            "aws iam attach-role-policy --role-name admin --policy-arn arn:aws:iam::aws:policy/AdministratorAccess",
            "aws iam detach-role-policy --role-name admin --policy-arn arn:aws:iam::aws:policy/AdministratorAccess",
        ],
    )
    def test_iam_mutating_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd), f"IAM op not blocked: {cmd!r}"


# ── Tier 7: IaC destructive ──────────────────────────────────


class TestTier7IaCDestructive:
    @pytest.mark.parametrize(
        "cmd",
        [
            "terraform destroy",
            "terraform destroy -auto-approve",
            "terraform destroy -target=aws_instance.web",
        ],
    )
    def test_terraform_destroy_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd), f"terraform destroy not blocked: {cmd!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            "terraform apply -auto-approve",
            "terraform apply -var='env=prod' -auto-approve",
        ],
    )
    def test_terraform_apply_auto_approve_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd), f"terraform apply -auto-approve not blocked: {cmd!r}"

    def test_pulumi_destroy_blocked(self, deny_patterns):
        assert _blocked(deny_patterns, "pulumi destroy")
        assert _blocked(deny_patterns, "pulumi destroy --yes")

    def test_cdk_destroy_blocked(self, deny_patterns):
        assert _blocked(deny_patterns, "cdk destroy")
        assert _blocked(deny_patterns, "cdk destroy MyStack")


# ── False positive prevention: read-only commands ─────────────


class TestReadOnlyCommandsNotBlocked:
    @pytest.mark.parametrize(
        "cmd",
        [
            "aws ecs list-tasks --cluster prod",
            "aws ecs describe-services --cluster prod --services web",
            "aws ecs describe-tasks --cluster prod --tasks abc",
            "aws ecs list-services --cluster prod",
            "aws ec2 describe-instances --instance-ids i-1234",
            "aws s3 ls s3://my-bucket/",
            "aws s3 cp s3://my-bucket/file.txt ./file.txt",
            "aws logs filter-log-events --log-group-name /ecs/app --start-time 1000",
            "aws sts get-caller-identity",
            "aws configure list",
            "aws configure list-profiles",
            "aws configure get region",
            "aws cloudformation describe-stacks --stack-name prod",
            "aws rds describe-db-instances --db-instance-identifier mydb",
            "aws iam list-users",
            "aws iam get-role --role-name admin",
            "terraform plan",
            "terraform init",
            "terraform fmt",
            "terraform validate",
            "pulumi preview",
            "pulumi up",
            "cdk synth",
            "cdk diff",
        ],
    )
    def test_readonly_not_blocked(self, deny_patterns, cmd):
        assert not _blocked(deny_patterns, cmd), f"Read-only command wrongly blocked: {cmd!r}"


# ── animaworks-tool passthrough ───────────────────────────────


class TestAnimaworksToolPassthrough:
    @pytest.mark.parametrize(
        "cmd",
        [
            "animaworks-tool aws_collector list_services",
            "animaworks-tool aws_collector describe_service my-cluster my-service",
            "animaworks-tool aws_collector get_task_logs my-cluster my-task",
        ],
    )
    def test_aws_collector_not_blocked(self, deny_patterns, cmd):
        assert not _blocked(deny_patterns, cmd), f"animaworks-tool wrongly blocked: {cmd!r}"


# ── rin incident scenario replay ──────────────────────────────


class TestRinIncidentScenarioReplay:
    """Verify all commands from the rin incident (2026-03-20/21) are blocked."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "aws ecs execute-command --cluster stg --task abc --command '/bin/bash -c python manage.py prewarm --date=2026-03-23'",
            "aws ecs execute-command --cluster stg --task abc --command '/bin/bash -c python manage.py prewarm --date=2026-03-21'",
            "aws ecs execute-command --cluster stg --task abc --command '/bin/bash -c python manage.py prewarm --sync'",
        ],
    )
    def test_rin_ecs_execute_command_blocked(self, deny_patterns, cmd):
        assert _blocked(deny_patterns, cmd)

    def test_rin_ecs_run_task_blocked(self, deny_patterns):
        cmd = 'aws ecs run-task --cluster stg --task-definition prewarm:3 --overrides \'{"containerOverrides":[{"name":"app","command":["python","manage.py","prewarm","--date=2026-03-23"]}]}\''
        assert _blocked(deny_patterns, cmd)

    def test_rin_sso_login_blocked(self, deny_patterns):
        assert _blocked(deny_patterns, "aws sso login --profile future_sync_stg")

    @pytest.mark.parametrize(
        "cmd",
        [
            "aws ecs describe-services --cluster stg --services my-svc",
            "aws logs filter-log-events --log-group-name /ecs/prewarm",
            "aws sts get-caller-identity",
        ],
    )
    def test_rin_readonly_not_blocked(self, deny_patterns, cmd):
        assert not _blocked(deny_patterns, cmd)


# ── DataTalks.Club scenario ──────────────────────────────────


class TestDataTalksClubScenario:
    """Verify terraform destroy is blocked (DataTalks.Club incident)."""

    def test_terraform_destroy_blocked(self, deny_patterns):
        assert _blocked(deny_patterns, "terraform destroy")
        assert _blocked(deny_patterns, "terraform destroy -auto-approve")

    def test_terraform_apply_auto_approve_blocked(self, deny_patterns):
        assert _blocked(deny_patterns, "terraform apply -auto-approve")

    def test_terraform_plan_allowed(self, deny_patterns):
        assert not _blocked(deny_patterns, "terraform plan")
        assert not _blocked(deny_patterns, "terraform plan -out=plan.tfplan")
