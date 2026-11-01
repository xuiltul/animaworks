from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.i18n.engine import EngineClient, TranslationResponse
from scripts.i18n.pipeline import Target, Translator, _langs_for_target, load_configuration
from scripts.i18n.protect import protect_text
from scripts.i18n.segment import split_markdown
from scripts.i18n.validate import ValidationError, validate_translation


def test_per_target_langs_include_zh_for_templates_only() -> None:
    configuration, targets, _glossary = load_configuration()
    defaults = configuration["defaults"]
    by_name = {target.name: target for target in targets}

    assert _langs_for_target(by_name["templates"], defaults, None) == ["en", "ko", "zh"]
    assert _langs_for_target(by_name["docs"], defaults, None) == ["en", "ko"]
    assert _langs_for_target(by_name["templates"], defaults, ["zh"]) == ["zh"]
    assert _langs_for_target(by_name["docs"], defaults, ["zh"]) == []


def test_engine_extra_body_is_merged_into_request(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    captured: dict[str, object] = {}

    class _Response:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    def _post(url: str, *, headers: dict[str, str], json: dict[str, object], timeout: float) -> _Response:
        captured.update(json)
        return _Response()

    monkeypatch.setattr(httpx, "post", _post)
    client = EngineClient(
        "local",
        {
            "kind": "openai",
            "base_url": "http://127.0.0.1:1/v1",
            "model": "m",
            "extra_body": {"chat_template_kwargs": {"thinking": False}},
        },
    )
    assert client.translate("system", "user").text == "ok"
    assert captured["chat_template_kwargs"] == {"thinking": False}
    assert captured["model"] == "m"


def test_local_engine_disables_thinking() -> None:
    configuration, _targets, _glossary = load_configuration()
    assert configuration["engines"]["local"]["extra_body"] == {"chat_template_kwargs": {"thinking": False}}


def test_target_langs_is_optional_and_overrides_defaults() -> None:
    target = Target(
        name="sample",
        src="sample/ja",
        dst="sample/{lang}",
        include=("**/*.md",),
        copy=(),
        header="none",
        langs=("en", "ko"),
    )
    defaults = {"langs": ["en", "ko", "zh"]}
    assert _langs_for_target(target, defaults, None) == ["en", "ko"]
    assert _langs_for_target(target, defaults, ["zh"]) == []


class EchoEngine:
    def __init__(self, transform=None) -> None:
        self.calls: list[str] = []
        self.transform = transform or (lambda text: text)

    def translate(self, _system: str, user: str) -> TranslationResponse:
        self.calls.append(user)
        return TranslationResponse(self.transform(user), input_tokens=17, output_tokens=11, model="fake-model")


def _translator(
    root: Path,
    target: Target,
    engine: EchoEngine,
    *,
    fallback: EchoEngine | None = None,
    manifest_name: str = "manifest.json",
) -> Translator:
    configuration = {
        "defaults": {
            "engine": "fake",
            "fallback_engine": "fallback" if fallback else "",
            "batch_chars": 6000,
            "concurrency": 3,
        },
        "engines": {
            "fake": {"kind": "openai", "model": "fake-model"},
            "fallback": {"kind": "openai", "model": "fallback-model"},
        },
    }
    engines = {"fake": engine}
    if fallback:
        engines["fallback"] = fallback
    return Translator(
        root=root,
        configuration=configuration,
        targets=[target],
        glossary=[],
        engines=engines,
        manifest_path=root / manifest_name,
    )


def _docs_target(*, links: tuple[tuple[str, str], ...] = ()) -> Target:
    return Target(
        name="docs",
        src="docs/ja",
        dst="docs/{lang}",
        include=("**/*.md",),
        copy=(),
        header="html_comment",
        link_rewrite=links,
    )


def test_protection_round_trip_preserves_code_placeholders_urls_and_frontmatter() -> None:
    source = (
        "---\n"
        "name: sample-skill\n"
        "description: >-\n"
        "  Use when: {anima_name} でpermissions.mdの処理をする。リンク [docs](https://example.test/a)。\n"
        "tags: [stable, cli]\n"
        "---\n\n"
        "値は {{literal}} と `{task.id}`。\n\n"
        "```python\nvalue = '{task_id}'\n```\n"
    )
    frontmatter, body = split_markdown(source)
    protected_frontmatter = protect_text(frontmatter.text, frontmatter=True)
    protected_body = protect_text(body.text)

    assert protected_frontmatter.text != frontmatter.text
    assert "Use when:" not in protected_frontmatter.text
    assert protected_frontmatter.restore(protected_frontmatter.text) == frontmatter.text
    assert protected_body.restore(protected_body.text) == body.text
    assert "name: sample-skill" in protected_frontmatter.restore(protected_frontmatter.text)
    assert "https://example.test/a" in protected_frontmatter.restore(protected_frontmatter.text)
    assert any(value == "permissions.md" for value in protected_frontmatter.values.values())
    assert "```python\nvalue = '{task_id}'\n```\n" in protected_body.values.values()


def test_protection_folds_sentinels_nested_by_a_later_pattern() -> None:
    # The path pattern spans placeholders that an earlier pass already replaced.
    source = "履歴は {animas_dir}/{name}/activity_log/{date}.jsonl で確認する。\n"
    protected = protect_text(source)

    visible = set(re.findall(r"⟦P\d+⟧", protected.text))
    assert visible == set(protected.values)
    assert not any(re.search(r"⟦P\d+⟧", value) for value in protected.values.values())
    assert protected.restore(protected.text) == source


def test_splitter_ignores_hash_lines_inside_fenced_code() -> None:
    from scripts.i18n.segment import split_markdown

    segments = split_markdown("# First\n\n```text\n# not a heading\n```\n\n## Second\nbody\n")

    assert len(segments) == 2
    assert "# not a heading" in segments[0].text
    assert segments[1].text.startswith("## Second")


def _make_ja_translator(tmp_path: Path) -> tuple[Translator, EchoEngine, Path]:
    """A docs translator whose mock engine turns Japanese fixtures into English."""
    translations = {
        "章A": "Section A",
        "アルファの説明。": "Alpha explanation.",
        "章B": "Section B",
        "ベータの説明。": "Beta explanation.",
        "ベータ改訂。": "Beta revised.",
        "章C": "Section C",
        "ガンマの説明。": "Gamma explanation.",
        "章D": "Section D",
        "デルタの説明。": "Delta explanation.",
    }

    def translate(text: str) -> str:
        for ja, en in translations.items():
            text = text.replace(ja, en)
        return text

    source_dir = tmp_path / "docs" / "ja"
    source_dir.mkdir(parents=True)
    source = source_dir / "guide.md"
    engine = EchoEngine(translate)
    translator = _translator(tmp_path, _docs_target(), engine)
    return translator, engine, source


def test_only_changed_section_is_translated_and_insertions_reuse_other_sections(tmp_path: Path) -> None:
    translator, engine, source = _make_ja_translator(tmp_path)
    source.write_text("# 章A\nアルファの説明。\n\n## 章B\nベータの説明。\n\n## 章C\nガンマの説明。\n", encoding="utf-8")

    assert translator.run("docs", requested_langs=["en"]) == 0
    first_call_count = len(engine.calls)
    source.write_text(
        "# 章A\nアルファの説明。\n\n## 章B\nベータ改訂。\n\n## 章C\nガンマの説明。\n",
        encoding="utf-8",
    )
    assert translator.run("docs", requested_langs=["en"]) == 0
    assert len(engine.calls) - first_call_count == 1
    assert "ベータ改訂。" in engine.calls[-1]
    assert "アルファの説明。" not in engine.calls[-1]
    assert "ガンマの説明。" not in engine.calls[-1]

    calls_before_insert = len(engine.calls)
    source.write_text(
        "# 章A\nアルファの説明。\n\n## 章B\nベータ改訂。\n\n## 章C\nガンマの説明。\n\n## 章D\nデルタの説明。\n",
        encoding="utf-8",
    )
    assert translator.run("docs", requested_langs=["en"]) == 0
    assert len(engine.calls) - calls_before_insert == 1
    assert engine.calls[-1].strip() == "## 章D\nデルタの説明。"


def test_malformed_batch_retries_each_section_only_once(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs" / "ja"
    source_dir.mkdir(parents=True)
    (source_dir / "guide.md").write_text(
        "# 第一節\nこれは説明です。\n\n## 第二節\nこれは補足です。\n", encoding="utf-8"
    )
    engine = EchoEngine(lambda _text: "malformed response")
    translator = _translator(tmp_path, _docs_target(), engine)

    assert translator.run("docs", requested_langs=["en"]) == 2
    # One malformed batch request, then exactly one single-section retry per section.
    assert len(engine.calls) == 3
    assert translator.manifest["files"]["docs/en/guide.md"]["statuses"] == ["failed", "failed"]


def test_invalid_single_section_is_retried_once_before_failure(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs" / "ja"
    source_dir.mkdir(parents=True)
    (source_dir / "guide.md").write_text("# 日本語\nこれは文章です。\n", encoding="utf-8")
    engine = EchoEngine()
    translator = _translator(tmp_path, _docs_target(), engine)

    assert translator.run("docs", requested_langs=["en"]) == 2
    assert len(engine.calls) == 2
    assert translator.manifest["files"]["docs/en/guide.md"]["statuses"] == ["failed"]


def test_failed_changed_section_keeps_its_previous_translation(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs" / "ja"
    source_dir.mkdir(parents=True)
    source = source_dir / "guide.md"
    source.write_text("# 見出し\n以前の文章です。\n", encoding="utf-8")
    engine = EchoEngine(lambda text: text.replace("見出し", "Heading").replace("以前の文章です。", "Previous text."))
    translator = _translator(tmp_path, _docs_target(), engine)

    assert translator.run("docs", requested_langs=["en"]) == 0
    source.write_text("# 見出し\n新しい文章です。\n", encoding="utf-8")
    engine.transform = lambda text: text

    assert translator.run("docs", requested_langs=["en"]) == 2
    output = (tmp_path / "docs" / "en" / "guide.md").read_text(encoding="utf-8")
    assert "Previous text." in output
    assert "新しい文章です。" not in output
    assert translator.manifest["files"]["docs/en/guide.md"]["statuses"] == ["failed"]


def test_lost_sentinel_fails_validation_and_uses_fallback(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs" / "ja"
    source_dir.mkdir(parents=True)
    (source_dir / "guide.md").write_text(
        "# セクション\nこれは説明です。\n\n```text\ncode stays byte-identical\n```\n",
        encoding="utf-8",
    )
    primary = EchoEngine(lambda text: re.sub(r"⟦P\d+⟧", "", text, count=1))
    fallback = EchoEngine(
        lambda text: text.replace("セクション", "Section").replace("これは説明です。", "This is an explanation.")
    )
    translator = _translator(tmp_path, _docs_target(), primary, fallback=fallback)

    assert translator.run("docs", requested_langs=["en"]) == 0
    assert primary.calls
    assert fallback.calls
    output = (tmp_path / "docs" / "en" / "guide.md").read_text(encoding="utf-8")
    assert "# Section" in output
    assert "This is an explanation." in output
    assert "```text\ncode stays byte-identical\n```" in output
    record = translator.manifest["files"]["docs/en/guide.md"]
    assert record["statuses"] == ["translated"]


def _compress_translate(translations):
    """Simulate a model that translates but drops the coalescing blank line
    between batch sections, so headings come back glued to the previous line."""

    def transform(text: str) -> str:
        for ja, en in translations.items():
            text = text.replace(ja, en)
        parts: list[str] = []
        for chunk in re.split(r"(⟦§\d+⟧)", text):
            if not chunk:
                continue
            if chunk.startswith("⟦§"):
                parts.append(chunk)
            else:
                parts.append(chunk.strip())
        return "".join(parts)

    return transform


def test_heading_stays_at_line_start_when_model_compresses_section_gaps(tmp_path: Path) -> None:
    # R22 regression: a model that strips the blank separator line between batch
    # sections glued the next heading onto the previous paragraph. Boundaries
    # must be restored from the source so each heading is at line start.
    source_dir = tmp_path / "docs" / "ja"
    source_dir.mkdir(parents=True)
    source = source_dir / "guide.md"
    source.write_text("# 章A\nアルファの説明。\n\n## 章B\nベータの説明。\n", encoding="utf-8")
    engine = EchoEngine(
        _compress_translate(
            {
                "章A": "Section A",
                "アルファの説明。": "Alpha explanation.",
                "章B": "Section B",
                "ベータの説明。": "Beta explanation.",
            }
        )
    )
    translator = _translator(tmp_path, _docs_target(), engine)

    assert translator.run("docs", requested_langs=["en"]) == 0
    output = (tmp_path / "docs" / "en" / "guide.md").read_text(encoding="utf-8")
    # Each heading is on its own line and the blank line gap matches the source
    # (the docs target prepends an auto-translation header).
    assert output.endswith("# Section A\nAlpha explanation.\n\n## Section B\nBeta explanation.\n")
    assert "\n\n## Section B" in output


def test_sections_without_translatable_japanese_are_kept_and_not_sent(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs" / "ja"
    source_dir.mkdir(parents=True)
    source = source_dir / "guide.md"
    source.write_text("# 日本語\nこれは説明。\n\n## ASCII Only\nKeep this verbatim 123.\n", encoding="utf-8")
    engine = EchoEngine(lambda text: text.replace("これは説明。", "This is an explanation."))
    translator = _translator(tmp_path, _docs_target(), engine)

    assert translator.run("docs", requested_langs=["en"]) == 0
    output = (tmp_path / "docs" / "en" / "guide.md").read_text(encoding="utf-8")
    assert "Keep this verbatim 123." in output
    # Only the Japanese section reached the model.
    assert len(engine.calls) == 1
    assert "ASCII Only" not in engine.calls[-1]


def test_validate_rejects_canned_model_response() -> None:
    source = "# 日本語\nこれは説明。\n"
    translated = (
        "Understood. Please provide the Japanese content you would like me to translate into Korean.\n"
        "# 日本語\nこれは説明。\n"
    )
    with pytest.raises(ValidationError, match="canned model response"):
        validate_translation(source, translated, "en")


def test_language_ratio_skipped_for_short_japanese_sections() -> None:
    # Fewer than 20 visible source characters: ratio thresholds are skipped, so
    # an English section that keeps a short Japanese example is accepted.
    source = "# 見出し\nここは短いです。\n"
    translated = "# Heading\nThis is short. これは例の文章です。\n"
    validate_translation(source, translated, "en")


def test_language_ratio_still_rejects_untranslated_short_section() -> None:
    source = "# 見出し\nここは短いです。\n"
    with pytest.raises(ValidationError, match="was not performed"):
        validate_translation(source, source, "en")


def test_language_ratio_fails_for_long_mostly_japanese_section() -> None:
    source = "# 見出し\n" + "これは長い文章です。" * 10 + "\n"
    with pytest.raises(ValidationError, match="was not performed"):
        validate_translation(source, source, "en")


def test_language_ratio_excludes_source_shared_japanese_for_korean() -> None:
    # Japanese usage examples that already appear in the source must not be
    # counted as untranslated kana in the Korean output.
    body = "ここに長い日本語の説明文が続きます。" * 8 + "\n"
    source = "# 日本語ガイド\n" + body + "「手順」の通りに操作します。\n"
    # Korean output keeps the same Japanese examples verbatim (as the source has
    # them) while translating the surrounding prose into Hangul.
    translated = "# 한국어 가이드\n여기에 긴 한국어 설명문이 계속됩니다." * 8 + "\n「手順」의 대로 조작합니다.\n"
    validate_translation(source, translated, "ko")


def test_url_check_stops_at_backticks_and_japanese_text() -> None:
    source = "# 接続\nOllama（`http://localhost:11434`）で接続先を指定します。\n"
    translated = "# Connection\nSpecify the endpoint with Ollama (`http://localhost:11434`).\n"
    validate_translation(source, translated, "en")


def test_template_frontmatter_that_is_not_yaml_is_accepted() -> None:
    source = "---\nname: {{skill_name}}\ndescription: >-\n  {{機能の説明}}\n---\n"
    translated = "---\nname: {{skill_name}}\ndescription: >-\n  {{capability summary}}\n---\n"
    validate_translation(source, translated, "en", frontmatter=True)


def test_language_ratio_accepts_chinese_han_characters() -> None:
    # Simplified Chinese shares Han characters with Japanese; only kana marks
    # an untranslated section.
    source = "# 記憶の仕組み\n" + "これは記憶システムの長い説明文です。" * 8 + "\n"
    translated = "# 记忆机制\n" + "这是关于记忆系统的详细说明文字。" * 8 + "\n"
    validate_translation(source, translated, "zh")


def test_language_ratio_rejects_untranslated_chinese_section() -> None:
    source = "# 見出し\n" + "これは長い文章です。" * 10 + "\n"
    with pytest.raises(ValidationError):
        validate_translation(source, source, "zh")


def test_language_ratio_rejects_korean_output_without_hangul() -> None:
    source = "# 見出し\n" + "これは長い文章です。" * 10 + "\n"
    translated = source  # stays Japanese (kana), no Hangul
    with pytest.raises(ValidationError):
        validate_translation(source, translated, "ko")


def test_check_detects_source_changes(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs" / "ja"
    source_dir.mkdir(parents=True)
    source = source_dir / "guide.md"
    source.write_text("# Guide\nSource text.\n", encoding="utf-8")
    translator = _translator(tmp_path, _docs_target(), EchoEngine())

    assert translator.run("docs", requested_langs=["en"]) == 0
    assert translator.run("docs", requested_langs=["en"], check=True) == 0
    source.write_text("# Guide\nChanged source text.\n", encoding="utf-8")
    assert translator.run("docs", requested_langs=["en"], check=True) == 1


def test_check_detects_generated_files_without_a_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs" / "ja"
    source_dir.mkdir(parents=True)
    (source_dir / "guide.md").write_text("# Guide\nSource text.\n", encoding="utf-8")
    translator = _translator(tmp_path, _docs_target(), EchoEngine())

    assert translator.run("docs", requested_langs=["en"]) == 0
    stale = tmp_path / "docs" / "en" / "retired.md"
    stale.write_text("# Retired\n", encoding="utf-8")

    assert translator.run("docs", requested_langs=["en"], check=True) == 1


def test_templates_have_no_headers_and_json_py_are_byte_copied(tmp_path: Path) -> None:
    source_dir = tmp_path / "templates" / "ja"
    source_dir.mkdir(parents=True)
    (source_dir / "guide.md").write_text("# Guide\nTemplate body.\n", encoding="utf-8")
    json_bytes = b'{"permissions": ["read"]}\n'
    python_bytes = b"VALUE = 7\n"
    (source_dir / "permissions.json").write_bytes(json_bytes)
    (source_dir / "helper.py").write_bytes(python_bytes)
    target = Target(
        name="templates",
        src="templates/ja",
        dst="templates/{lang}",
        include=("**/*.md",),
        copy=("**/*.json", "**/*.py"),
        header="none",
    )
    translator = _translator(tmp_path, target, EchoEngine())

    assert translator.run("templates", requested_langs=["en"]) == 0
    output_dir = tmp_path / "templates" / "en"
    assert (output_dir / "guide.md").read_text(encoding="utf-8") == "# Guide\nTemplate body.\n"
    assert not (output_dir / "guide.md").read_text(encoding="utf-8").startswith("<!-- 自動翻訳")
    assert (output_dir / "permissions.json").read_bytes() == json_bytes
    assert (output_dir / "helper.py").read_bytes() == python_bytes
    assert translator.manifest["files"]["templates/en/guide.md"]["source_sha256"]
    assert translator.run("templates", requested_langs=["en"], check=True) == 0


def test_readme_rewrites_only_link_destinations_for_each_language(tmp_path: Path) -> None:
    source = tmp_path / "README_ja.md"
    source.write_text(
        "# README\nSee [guide](docs/ja/reference/api.md) and [other](README_ja.md).\n"
        "\n```md\n[example](docs/ja/not-a-link.md)\n```\n",
        encoding="utf-8",
    )
    target = Target(
        name="readme",
        src="README_ja.md",
        dst={"en": "README.md", "ko": "README_ko.md"},
        include=("README_ja.md",),
        copy=(),
        header="html_comment",
        link_rewrite=(("docs/ja/", "docs/{lang}/"), ("README_ja.md", "{readme}")),
    )
    translator = _translator(tmp_path, target, EchoEngine())

    assert translator.run("readme", requested_langs=["en", "ko"]) == 0
    english = (tmp_path / "README.md").read_text(encoding="utf-8")
    korean = (tmp_path / "README_ko.md").read_text(encoding="utf-8")
    for output, readme in ((english, "README.md"), (korean, "README_ko.md")):
        assert "](docs/" + ("en" if readme == "README.md" else "ko") + "/reference/api.md)" in output
        assert f"]({readme})" in output
        assert "[example](docs/ja/not-a-link.md)" in output
        assert "source-sha256=" in output.splitlines()[1]
