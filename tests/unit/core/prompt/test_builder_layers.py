from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.memory.manager import MemoryManager
from core.prompt import builder


def _memory(anima_dir: Path, *, vision: str = "") -> MagicMock:
    for child in ("knowledge", "procedures", "skills"):
        (anima_dir / child).mkdir(parents=True, exist_ok=True)

    memory = MagicMock()
    memory.anima_dir = anima_dir
    memory.read_identity.return_value = "# Identity\nFixture anima"
    memory.read_injection.return_value = ""
    memory.read_permissions.return_value = ""
    memory.read_bootstrap.return_value = ""
    memory.read_company_vision.return_value = vision
    memory.read_specialty_prompt.return_value = ""
    memory.read_current_state.return_value = "status: idle"
    memory.read_resolutions.return_value = []
    memory.list_knowledge_files.return_value = []
    memory.list_procedure_files.return_value = []
    memory.list_shared_users.return_value = []
    return memory


def _build(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    trigger: str,
    *,
    anima_name: str = "fixture",
) -> str:
    anima_dir = data_dir / "animas" / anima_name
    memory = _memory(anima_dir)
    monkeypatch.setattr(builder, "get_data_dir", lambda: data_dir)
    monkeypatch.setattr(builder, "_discover_other_animas", lambda _path: [])
    return builder.build_system_prompt(
        memory,
        execution_mode="s",
        trigger=trigger,
        context_window=200_000,
    ).system_prompt


def test_trigger_specific_behavior_context(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chat = _build(data_dir, monkeypatch, "chat")
    heartbeat = _build(data_dir, monkeypatch, "heartbeat")
    task = _build(data_dir, monkeypatch, "task:fixture")

    assert "業務指示を受けた場合の振り分け" in chat
    assert "チャットでのタスク記録" in chat
    assert "Heartbeat でのタスク記録" not in chat

    assert "業務指示を受けた場合の振り分け" not in heartbeat
    assert "Heartbeat でのタスク記録" in heartbeat
    assert "チャットでのタスク記録" not in heartbeat

    assert "業務指示を受けた場合の振り分け" not in task
    assert "チャットでのタスク記録" not in task
    assert "Heartbeat でのタスク記録" not in task


def test_repo_rules_only_with_workspace(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder, "_read_default_workspace", lambda _path: "")
    without_workspace = _build(data_dir, monkeypatch, "task:fixture")

    monkeypatch.setattr(builder, "_read_default_workspace", lambda _path: "WORKSPACE_MARKER")
    with_workspace = _build(data_dir, monkeypatch, "task:fixture")

    assert "リポジトリ作業ルール" not in without_workspace
    assert "WORKSPACE_MARKER" in with_workspace
    assert "リポジトリ作業ルール" in with_workspace


def test_environment_is_l1_and_points_to_reference(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = _build(data_dir, monkeypatch, "chat")

    assert "├──" not in prompt
    assert 'read_memory_file(path="reference/anatomy/environment-layout.md")' in prompt


@pytest.mark.parametrize("trigger", ["chat", "heartbeat", "task:fixture", "inbox:peer"])
def test_framework_resident_prompt_budget(data_dir: Path, monkeypatch: pytest.MonkeyPatch, trigger: str) -> None:
    from core.prompt.tokens import estimate_tokens

    prompt = _build(data_dir, monkeypatch, trigger)
    count = estimate_tokens(prompt)
    print(f"framework_prompt {trigger} tokens={count}")
    assert count <= 6000
    assert "Fixture anima" in prompt
    assert "回答の前に記憶を確認せよ" not in prompt


def test_large_recall_does_not_evict_resident_rules_tasks_or_human_notifications(data_dir: Path, monkeypatch):
    from core.memory.priming.format import format_priming_section
    from core.memory.priming.result import PrimingResult

    memory = _memory(data_dir / "animas" / "fixture")
    memory.read_injection.return_value = "ROLE_SAFETY: external sending requires human approval."
    monkeypatch.setattr(builder, "get_data_dir", lambda: data_dir)
    monkeypatch.setattr(builder, "_discover_other_animas", lambda _path: [])
    monkeypatch.setattr(
        builder, "_build_resolved_approvals_section", lambda *_args: "APPROVAL_STATE: decision settled."
    )
    priming = PrimingResult(
        resident_knowledge='MANDATORY_POLICY → read_memory_file(path="knowledge/approval-policy.md")',
        pending_tasks="PENDING_REQUEST: wait for customer approval",
        recent_outbound="DUPLICATE_GUARD: invoice already sent",
        related_knowledge_untrusted="UNTRUSTED_RECALL " * 12_000,
    )
    prompt = builder.build_system_prompt(
        memory,
        execution_mode="s",
        trigger="chat",
        context_window=200_000,
        priming_section=format_priming_section(priming),
        pending_human_notifications="HUMAN_NOTIFICATION: outstanding decision",
    ).system_prompt
    for preserved in (
        "ROLE_SAFETY",
        "MANDATORY_POLICY",
        "approval-policy.md",
        "PENDING_REQUEST",
        "DUPLICATE_GUARD",
        "HUMAN_NOTIFICATION",
        "APPROVAL_STATE",
    ):
        assert preserved in prompt
    assert "UNTRUSTED_RECALL" not in prompt
    assert prompt.count("<priming ") == prompt.count("</priming>")


@pytest.mark.parametrize("execution_mode", ["a", "c"])
def test_modest_recall_pointers_survive_full_framework_prompt(data_dir: Path, monkeypatch, execution_mode):
    from core.memory.priming.format import format_priming_section
    from core.memory.priming.result import PrimingResult
    from core.prompt.tokens import estimate_tokens

    memory = _memory(data_dir / "animas" / "fixture")
    monkeypatch.setattr(builder, "get_data_dir", lambda: data_dir)
    monkeypatch.setattr(builder, "_discover_other_animas", lambda _path: [])
    pointers = [f'CUSTOMER_{i:02d} → read_memory_file(path="knowledge/customer-{i:02d}.md")' for i in range(12)]
    result = builder.build_system_prompt(
        memory,
        execution_mode=execution_mode,
        trigger="chat",
        context_window=200_000,
        priming_section=format_priming_section(PrimingResult(related_knowledge="\n".join(pointers))),
    )
    for index in range(12):
        assert f"customer-{index:02d}.md" in result.system_prompt
    assert estimate_tokens(result.system_prompt) <= 8000


def test_permission_source_survives_even_an_infeasible_context_ceiling(data_dir: Path, monkeypatch):
    memory = _memory(data_dir / "animas" / "fixture")
    permissions = "PERMISSION_ORIGINAL: never send without explicit approval. " * 400
    memory.read_permissions.return_value = permissions
    monkeypatch.setattr(builder, "get_data_dir", lambda: data_dir)
    monkeypatch.setattr(builder, "_discover_other_animas", lambda _path: [])
    result = builder.build_system_prompt(memory, execution_mode="c", trigger="chat", context_window=4000)
    assert permissions in result.system_prompt


def test_cli_duplication_and_skill_creator_are_removed(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = _build(data_dir, monkeypatch, "chat")

    assert "## CLI Tools" not in prompt
    assert prompt.count("skill-creator") <= 1


def test_communication_rules_are_injected_once(
    data_dir: Path,
    make_anima,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = make_anima("sakura")
    make_anima("rin", supervisor="sakura", speciality="development")
    memory = _memory(anima_dir)
    monkeypatch.setattr(builder, "get_data_dir", lambda: data_dir)

    prompt = builder.build_system_prompt(memory, execution_mode="s", trigger="chat").system_prompt

    assert prompt.count("**経路**:") == 1
    assert prompt.count('`ping_subordinate(name="<Anima名>")`') == 1


def test_read_identity_strips_frontmatter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    memory = MemoryManager.__new__(MemoryManager)
    memory.anima_dir = tmp_path
    raw = "---\nname: Fixture\nrole: engineer\n---\n\n# Identity\nFixture anima"
    monkeypatch.setattr(memory, "_read", lambda _path: raw)

    assert memory.read_identity() == "# Identity\nFixture anima"


@pytest.mark.parametrize("vision", ["# Vision\n要記入", "# Vision\nTODO", "# Vision\n(未記入)"])
def test_placeholder_vision_is_not_injected(vision: str) -> None:
    memory = MagicMock()
    memory.read_bootstrap.return_value = ""
    memory.read_company_vision.return_value = vision
    memory.read_specialty_prompt.return_value = ""

    sections = builder._build_group2(memory, "", False, False, {})

    assert "vision" not in {section.id for section in sections}


def test_substantive_vision_is_injected() -> None:
    vision = "# Vision\nBuild reliable systems that measurably improve how the organization works every day."
    memory = MagicMock()
    memory.read_bootstrap.return_value = ""
    memory.read_company_vision.return_value = vision
    memory.read_specialty_prompt.return_value = ""

    sections = builder._build_group2(memory, "", False, False, {})

    assert next(section.content for section in sections if section.id == "vision") == vision
