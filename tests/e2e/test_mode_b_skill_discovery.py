from __future__ import annotations

import re

import pytest

from core.prompt.tokens import estimate_tokens
from tests.helpers.mocks import make_litellm_response, patch_litellm


@pytest.mark.timeout(20)
async def test_mode_b_preserves_query_matched_skill_under_default_budget(make_agent_core, monkeypatch):
    agent = make_agent_core(name="b-ranked-skills", model="ollama/gemma3:27b", execution_mode="assisted")

    def write_skill(name: str, description: str, extra: str = "") -> None:
        path = agent.anima_dir / "skills" / name / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(
            f"---\nname: {name}\ndescription: {description}\n{extra}---\n# Fixture\nRead synthetic evidence."
        )

    for index in range(30):
        write_skill(f"aaa-filler-{index:02}", "Unrelated gardening notes.")
    write_skill(
        "zzz-release-audit", "Verify quasar release manifest.", "trigger_phrases: [verify quasar release manifest]\n"
    )
    write_skill(
        "zzz-blocked-audit",
        "Verify quasar release manifest.",
        "trust_level: blocked\ntrigger_phrases: [verify quasar release manifest]\n",
    )
    # Exercise deterministic query routing, not embedding quality.
    monkeypatch.setattr("core.skills.dense.skill_dense_scores", lambda *args, **kwargs: {})
    captured = []
    original_call = agent._executor._call_llm

    async def capture_call(messages, **kwargs):
        captured.extend(message["content"] for message in messages if message.get("role") == "system")
        return await original_call(messages, **kwargs)

    monkeypatch.setattr(agent._executor, "_call_llm", capture_call)
    with patch_litellm(make_litellm_response(content="Synthetic response.")):
        await agent.run_cycle("Please verify quasar release manifest.")

    assert captured
    catalog = re.search(r'<section name="skill_catalog">\n(.*?)\n</section>', captured[0], re.DOTALL)
    assert catalog is not None
    assert "skills/zzz-release-audit/SKILL.md" in catalog.group(1)
    assert "Verify quasar release manifest." in catalog.group(1)
    assert "zzz-blocked-audit" not in captured[0]
    assert estimate_tokens(catalog.group(1)) <= 512
