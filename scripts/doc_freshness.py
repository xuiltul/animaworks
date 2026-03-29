#!/usr/bin/env python3
"""Document freshness checker & auto-updater.

Detects stale documents by comparing git timestamps of template docs
against their related source code, and optionally auto-updates them
via cursor-agent.

Usage:
    python scripts/doc_freshness.py                    # Report stale docs
    python scripts/doc_freshness.py --json             # JSON output
    python scripts/doc_freshness.py --fix              # Auto-update via cursor-agent
    python scripts/doc_freshness.py --fix --dry-run    # Preview fix commands
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────

WORKSPACE = Path(__file__).resolve().parent.parent
TEMPLATES = WORKSPACE / "templates"

_HIGH_DAYS = 7
_MED_DAYS = 3

_SEV_LABEL = {"HIGH": "HIGH", "MEDIUM": "MED ", "LOW": "LOW "}

CATEGORIES = (
    "common_skills",
    "common_knowledge",
    "docs",
    "root",
)

# ── Root-level files (not in templates/ or docs/) ─────────

_ROOT_FILE_MAP: dict[str, tuple[str, str]] = {
    "root/README": ("README_ja.md", "README.md"),
}

# ── ja → en path overrides (common_skills differ in structure) ───────

_SKILL_JA_TO_EN: dict[str, str] = {
    "common_skills/animaworks-guide/SKILL.md": "common_skills/animaworks_guide.md",
    "common_skills/cron-management/SKILL.md": "common_skills/cron-management.md",
    "common_skills/image-posting/SKILL.md": "common_skills/image-posting.md",
    "common_skills/skill-creator/SKILL.md": "common_skills/skill-creator.md",
    "common_skills/subagent-cli/SKILL.md": "common_skills/subagent-cli.md",
    "common_skills/subordinate-management/SKILL.md": ("common_skills/subordinate_management.md"),
    "common_skills/tool-creator/SKILL.md": "common_skills/tool-creator.md",
}

_JA_ONLY: frozenset[str] = frozenset(
    {
        "common_skills/skill-creator/references/description_guide.md",
        "common_skills/skill-creator/templates/skill_template.md",
    }
)

# ── DOC_SOURCE_MAP ───────────────────────────────────────────────────
# key  = relative path within locale dir (ja canonical)
# value = related source code paths (relative to workspace root)

DOC_SOURCE_MAP: dict[str, list[str]] = {
    # ── common_knowledge — communication ──
    "common_knowledge/communication/messaging-guide.md": [
        "core/messenger.py",
        "core/outbound.py",
    ],
    "common_knowledge/communication/board-guide.md": [
        "core/messenger.py",
        "server/routes/channels.py",
    ],
    "common_knowledge/communication/instruction-patterns.md": [
        "core/messenger.py",
        "core/tooling/handler_comms.py",
    ],
    "common_knowledge/communication/reporting-guide.md": [
        "core/messenger.py",
        "core/outbound.py",
    ],
    "common_knowledge/communication/sending-limits.md": [
        "core/outbound.py",
        "core/memory/activity.py",
    ],
    # ── common_knowledge — operations ──
    "common_knowledge/operations/task-management.md": [
        "core/background.py",
        "core/tooling/",
        "core/memory/task_queue.py",
    ],
    "common_knowledge/operations/heartbeat-cron-guide.md": [
        "core/background.py",
        "core/schedule_parser.py",
        "core/_anima_heartbeat.py",
    ],
    "common_knowledge/operations/background-tasks.md": [
        "core/background.py",
        "core/supervisor/pending_executor.py",
    ],
    "common_knowledge/operations/mode-s-auth-guide.md": [
        "core/execution/agent_sdk.py",
        "core/execution/_sdk_security.py",
    ],
    "common_knowledge/operations/project-setup.md": [
        "core/init.py",
        "cli/commands/init_cmd.py",
    ],
    "common_knowledge/operations/tool-usage-overview.md": [
        "core/tooling/",
        "core/tools/",
    ],
    "common_knowledge/operations/voice-chat-guide.md": [
        "core/voice/",
        "server/routes/voice.py",
    ],
    # ── common_knowledge — anatomy ──
    "common_knowledge/anatomy/what-is-anima.md": [
        "core/anima.py",
        "core/background.py",
        "core/lifecycle.py",
    ],
    "common_knowledge/anatomy/anima-anatomy.md": [
        "core/anima_factory.py",
        "core/init.py",
        "core/prompt/builder.py",
        "core/schedule_parser.py",
    ],
    "common_knowledge/anatomy/memory-system.md": [
        "core/memory/",
        "core/memory/priming.py",
        "core/memory/consolidation.py",
        "core/memory/forgetting.py",
        "core/memory/rag/",
    ],
    # ── common_knowledge — organization ──
    "common_knowledge/organization/hierarchy-rules.md": [
        "core/tooling/handler_org.py",
        "core/tooling/schemas.py",
    ],
    "common_knowledge/organization/roles.md": [
        "templates/_shared/",
        "core/anima_factory.py",
    ],
    "common_knowledge/organization/structure.md": [
        "core/org_sync.py",
        "core/anima_factory.py",
    ],
    # ── common_knowledge — security ──
    "common_knowledge/security/prompt-injection-awareness.md": [
        "core/prompt/builder.py",
        "core/execution/_sanitize.py",
    ],
    # ── common_knowledge — troubleshooting ──
    "common_knowledge/troubleshooting/common-issues.md": ["core/"],
    "common_knowledge/troubleshooting/escalation-flowchart.md": [
        "core/notification/",
        "core/tooling/handler_comms.py",
    ],
    # ── common_skills ──
    "common_skills/animaworks-guide/SKILL.md": [
        "cli/",
        "core/config/",
        "core/memory/",
    ],
    "common_skills/cron-management/SKILL.md": [
        "core/schedule_parser.py",
        "core/background.py",
    ],
    "common_skills/image-posting/SKILL.md": [
        "core/tools/image_gen.py",
        "core/image_artifacts.py",
    ],
    "common_skills/skill-creator/SKILL.md": [
        "core/tooling/skill_creator.py",
        "core/memory/skill_metadata.py",
    ],
    "common_skills/skill-creator/references/description_guide.md": [
        "core/memory/skill_metadata.py",
    ],
    "common_skills/skill-creator/templates/skill_template.md": [
        "core/tooling/skill_creator.py",
    ],
    "common_skills/subagent-cli/SKILL.md": [
        "core/execution/agent_sdk.py",
        "core/execution/codex_sdk.py",
    ],
    "common_skills/subordinate-management/SKILL.md": [
        "core/tooling/handler_org.py",
        "core/tooling/schemas.py",
    ],
    "common_skills/tool-creator/SKILL.md": ["core/tooling/", "core/tools/"],
    "common_skills/discord-tool/SKILL.md": [
        "core/tools/discord.py",
        "core/tools/_discord_client.py",
        "core/tools/_discord_cli.py",
    ],
    # ── docs (OSS公開リポジトリ同期ドキュメント) ──
    # publish.sh で公開リポジトリに同期されるファイル。
    # doc_key は "docs/{stem}" 形式。ja=docs/{stem}.ja.md, en=docs/{stem}.md
    "docs/vision": ["core/"],
    "docs/spec": ["core/", "server/"],
    "docs/features": ["core/", "server/", "cli/"],
    "docs/memory": ["core/memory/"],
    "docs/brain-mapping": ["core/memory/", "core/prompt/"],
    "docs/security": [
        "core/execution/_sanitize.py",
        "core/execution/_sdk_security.py",
        "core/tooling/handler.py",
        "core/prompt/builder.py",
    ],
    "docs/slack-socket-mode-setup": [
        "core/tools/slack.py",
        "server/routes/webhooks.py",
    ],
    # ── root-level files ──
    "root/README": [
        "core/",
        "core/execution/",
        "core/voice/",
        "core/memory/priming.py",
        "server/",
        "cli/",
        "templates/roles/",
    ],
}

# 対象は common_skills, common_knowledge, および OSS公開 docs/ 。
# 以下はすべてマッピング対象外:
# - prompts/ : システムプロンプトの一部。自動更新はリスク大
# - roles/ : permissions.json（設定テンプレート）
#            + specialty_prompt.md（プロンプトフラグメント）。prompts/と同種
# - anima_templates/_blank/* : Anima作成時にコピーされるスケルトン
# - bootstrap.md, company/vision.md : ユーザー編集前提の初期テンプレート
# - common_knowledge/00_index.md : ナビゲーションインデックス
# - docs/paper/ : 評価データ（コード変更追従不要）


# ── Data ─────────────────────────────────────────────────────────────


@dataclass
class StaleEntry:
    doc_key: str
    locale: str
    doc_path: str
    doc_date: str
    code_date: str
    changed_code: str
    severity: str
    days_stale: int


# ── Git helpers ──────────────────────────────────────────────────────

_ts_cache: dict[str, int | None] = {}


def _git_timestamp(rel_path: str) -> int | None:
    """Last commit timestamp (epoch) for a file or directory."""
    if rel_path in _ts_cache:
        return _ts_cache[rel_path]
    try:
        proc = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", rel_path],
            capture_output=True,
            text=True,
            cwd=WORKSPACE,
            timeout=10,
        )
        ts = int(proc.stdout.strip()) if proc.stdout.strip() else None
    except (subprocess.TimeoutExpired, ValueError):
        ts = None
    _ts_cache[rel_path] = ts
    return ts


def _ts_to_date(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")


# ── Path resolution ──────────────────────────────────────────────────


def _en_rel(doc_key: str) -> str | None:
    """Resolve en-locale relative path for a ja-canonical doc key."""
    if doc_key in _JA_ONLY:
        return None
    if doc_key in _ROOT_FILE_MAP:
        _, en = _ROOT_FILE_MAP[doc_key]
        return en if (WORKSPACE / en).exists() else None
    if doc_key.startswith("docs/"):
        stem = doc_key[len("docs/") :]
        rel = f"docs/{stem}.md"
        return rel if (WORKSPACE / rel).exists() else None
    en_key = _SKILL_JA_TO_EN.get(doc_key, doc_key)
    en_path = TEMPLATES / "en" / en_key
    return f"templates/en/{en_key}" if en_path.exists() else None


def _ja_rel(doc_key: str) -> str:
    if doc_key in _ROOT_FILE_MAP:
        ja, _ = _ROOT_FILE_MAP[doc_key]
        return ja
    if doc_key.startswith("docs/"):
        stem = doc_key[len("docs/") :]
        return f"docs/{stem}.ja.md"
    return f"templates/ja/{doc_key}"


# ── Staleness detection ──────────────────────────────────────────────


def _severity(days: int) -> str:
    if days >= _HIGH_DAYS:
        return "HIGH"
    if days >= _MED_DAYS:
        return "MEDIUM"
    return "LOW"


def _check_one(
    doc_key: str,
    code_paths: list[str],
    since_ts: int | None = None,
) -> list[StaleEntry]:
    """Check staleness of a single doc key across both locales."""
    results: list[StaleEntry] = []

    for locale in ("ja", "en"):
        doc_rel = _ja_rel(doc_key) if locale == "ja" else _en_rel(doc_key)
        if doc_rel is None:
            continue
        if not (WORKSPACE / doc_rel).exists():
            continue

        doc_ts = since_ts if since_ts is not None else _git_timestamp(doc_rel)
        if doc_ts is None:
            continue

        best_ts: int | None = None
        best_path = ""
        for cp in code_paths:
            cts = _git_timestamp(cp)
            if cts is not None and (best_ts is None or cts > best_ts):
                best_ts = cts
                best_path = cp

        if best_ts is None or best_ts <= doc_ts:
            continue

        days = (best_ts - doc_ts) // 86400
        if days < 1:
            continue

        results.append(
            StaleEntry(
                doc_key=doc_key,
                locale=locale,
                doc_path=doc_rel,
                doc_date=_ts_to_date(doc_ts),
                code_date=_ts_to_date(best_ts),
                changed_code=best_path,
                severity=_severity(days),
                days_stale=days,
            )
        )

    return results


# ── Discovery (--all) ────────────────────────────────────────────────


_PUBLISH_EXCLUDED_DIRS = frozenset(
    {
        "implemented",
        "issues",
        "research",
        "records",
        "reports",
        "drafts",
        "legacy",
        "testing",
    }
)
_PUBLISH_EXCLUDED_FILES = frozenset(
    {
        "index.md",
        "oss-publication-strategy.md",
    }
)


def _discover_unmapped() -> dict[str, list[str]]:
    """Find template/docs not in DOC_SOURCE_MAP; default code path = core/."""
    extra: dict[str, list[str]] = {}
    ja_dir = TEMPLATES / "ja"
    if ja_dir.is_dir():
        for md in ja_dir.rglob("*.md"):
            key = str(md.relative_to(ja_dir))
            if key not in DOC_SOURCE_MAP:
                extra[key] = ["core/"]
    docs_dir = WORKSPACE / "docs"
    if docs_dir.is_dir():
        for md in docs_dir.rglob("*.ja.md"):
            rel = md.relative_to(docs_dir)
            if rel.parts[0] in _PUBLISH_EXCLUDED_DIRS:
                continue
            stem = str(rel).removesuffix(".ja.md")
            key = f"docs/{stem}"
            if key not in DOC_SOURCE_MAP:
                extra[key] = ["core/"]
    return extra


# ── Collection ───────────────────────────────────────────────────────


def collect(
    *,
    all_docs: bool = False,
    file_filter: str | None = None,
    category: str | None = None,
    since: str | None = None,
) -> tuple[list[StaleEntry], int]:
    """Return (stale_entries, total_doc_paths_checked)."""
    since_ts: int | None = None
    if since:
        since_ts = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())

    doc_map: dict[str, list[str]] = dict(DOC_SOURCE_MAP)
    if all_docs:
        for k, v in _discover_unmapped().items():
            doc_map.setdefault(k, v)

    if file_filter:
        norm = file_filter
        for prefix in ("templates/ja/", "templates/en/"):
            if norm.startswith(prefix):
                norm = norm[len(prefix) :]
                break
        doc_map = {k: v for k, v in doc_map.items() if k == norm}

    if category:
        doc_map = {k: v for k, v in doc_map.items() if k.startswith(f"{category}/") or k.startswith(category)}

    total = 0
    entries: list[StaleEntry] = []
    for doc_key, code_paths in sorted(doc_map.items()):
        ja_exists = (WORKSPACE / _ja_rel(doc_key)).exists()
        en_path = _en_rel(doc_key)
        en_exists = en_path is not None and (WORKSPACE / en_path).exists()
        total += int(ja_exists) + int(en_exists)
        entries.extend(_check_one(doc_key, code_paths, since_ts))

    sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    entries.sort(key=lambda e: (sev_order[e.severity], -e.days_stale))

    return entries, total


# ── Report output ────────────────────────────────────────────────────


def print_report(entries: list[StaleEntry], total: int) -> None:
    print(f"\n=== Stale Documents: {len(entries)}/{total} ===\n")
    if not entries:
        print("  All documents are up to date.")
        return
    for e in entries:
        print(f"[{_SEV_LABEL[e.severity]}] {e.doc_path}")
        print(f"       Reason: {e.changed_code} changed {e.code_date} (doc: {e.doc_date}, {e.days_stale}d stale)")


def print_json_report(entries: list[StaleEntry]) -> None:
    data = [
        {
            "doc_key": e.doc_key,
            "locale": e.locale,
            "doc_path": e.doc_path,
            "severity": e.severity,
            "days_stale": e.days_stale,
            "doc_date": e.doc_date,
            "code_date": e.code_date,
            "changed_code": e.changed_code,
        }
        for e in entries
    ]
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ── Fix mode ─────────────────────────────────────────────────────────


def _agent_base_cmd(model: str) -> list[str]:
    """Build cursor-agent base command with required flags."""
    return [
        "cursor-agent",
        "-p",
        "--trust",
        "--force",
        "--workspace",
        str(WORKSPACE),
        "--model",
        model,
    ]


def run_fix(
    entries: list[StaleEntry],
    *,
    model: str,
    dry_run: bool = False,
    skip_en: bool = False,
    timeout: int = 1800,
) -> None:
    if not dry_run:
        agent = shutil.which("cursor-agent")
        if not agent:
            print(
                "Error: cursor-agent not found in PATH. Install it or use --dry-run to preview commands.",
                file=sys.stderr,
            )
            sys.exit(1)

    ja_entries = [e for e in entries if e.locale == "ja"]
    if not ja_entries:
        print("No stale ja documents to fix.")
        return

    fixed_keys: set[str] = set()

    for e in ja_entries:
        prompt = (
            f"{e.doc_path}が古くなっています。"
            f"関連コード({e.changed_code})の最新の変更を調査して、"
            f"ドキュメントを最新の実装に合わせて更新してください。"
        )
        cmd = [*_agent_base_cmd(model), prompt]

        if dry_run:
            print(f"[DRY-RUN] {_shell_join(cmd)}")
        else:
            print(f"\n>>> Fixing {e.doc_path} ...")
            try:
                result = subprocess.run(cmd, cwd=WORKSPACE, timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"  WARN: timed out after {timeout}s, skipping")
                continue
            if result.returncode != 0:
                print(f"  WARN: cursor-agent exited {result.returncode}, skipping")
                continue
        fixed_keys.add(e.doc_key)

    if skip_en:
        return

    for doc_key in sorted(fixed_keys):
        en_path = _en_rel(doc_key)
        if en_path is None:
            continue

        ja_path = _ja_rel(doc_key)
        prompt = (
            f"{ja_path}の内容を英語に翻訳して{en_path}を更新してください。"
            f"技術用語は適切に翻訳し、マークダウン構造は維持してください。"
        )
        cmd = [*_agent_base_cmd(model), prompt]

        if dry_run:
            print(f"[DRY-RUN] {_shell_join(cmd)}")
        else:
            print(f"\n>>> Translating to {en_path} ...")
            try:
                result = subprocess.run(cmd, cwd=WORKSPACE, timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"  WARN: timed out after {timeout}s, skipping")
                continue
            if result.returncode != 0:
                print(f"  WARN: cursor-agent exited {result.returncode}, skipping")


def _shell_join(cmd: list[str]) -> str:
    """Best-effort shell quoting for display."""
    import shlex

    return shlex.join(cmd)


# ── CLI ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Check document freshness and auto-update stale docs",
    )
    p.add_argument(
        "--fix",
        action="store_true",
        help="Auto-update stale docs via cursor-agent",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview fix commands without executing",
    )
    p.add_argument(
        "--model",
        default="composer-2",
        help="Model for cursor-agent (default: composer-2)",
    )
    p.add_argument(
        "--all",
        dest="all_docs",
        action="store_true",
        help="Include unmapped docs (compare against core/ by default)",
    )
    p.add_argument(
        "--file",
        dest="file_filter",
        metavar="PATH",
        help="Check a specific document file",
    )
    p.add_argument(
        "--category",
        choices=list(CATEGORIES),
        help="Filter by template category",
    )
    p.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        help="Override doc date (treat all docs as last updated on this date)",
    )
    p.add_argument(
        "--skip-en",
        action="store_true",
        help="Skip en translation in fix mode",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=1800,
        metavar="SEC",
        help="Per-file timeout in seconds for cursor-agent (default: 1800)",
    )
    p.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="JSON output for tooling integration",
    )
    return p


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    args = build_parser().parse_args()

    entries, total = collect(
        all_docs=args.all_docs,
        file_filter=args.file_filter,
        category=args.category,
        since=args.since,
    )

    if args.fix or args.dry_run:
        if not entries:
            print("No stale documents found.")
            return
        run_fix(
            entries,
            model=args.model,
            dry_run=args.dry_run,
            skip_en=args.skip_en,
            timeout=args.timeout,
        )
    elif args.json_output:
        print_json_report(entries)
    else:
        print_report(entries, total)


if __name__ == "__main__":
    main()
