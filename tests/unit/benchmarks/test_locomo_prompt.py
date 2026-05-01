"""Tests for LoCoMo benchmark answer prompt improvements (Issue #174)."""

from __future__ import annotations


class TestAnswerPromptConstants:
    """Verify answer prompt templates contain required elements."""

    def test_neo4j_adapter_has_answer_system(self) -> None:
        from benchmarks.locomo.neo4j_adapter import _ANSWER_SYSTEM

        assert "expert assistant" in _ANSWER_SYSTEM
        assert "past conversations" in _ANSWER_SYSTEM

    def test_neo4j_adapter_has_answer_template(self) -> None:
        from benchmarks.locomo.neo4j_adapter import _ANSWER_TEMPLATE

        assert "{context}" in _ANSWER_TEMPLATE
        assert "{question}" in _ANSWER_TEMPLATE
        assert "event_time" in _ANSWER_TEMPLATE
        assert "7 May 2023" in _ANSWER_TEMPLATE

    def test_legacy_adapter_has_answer_system(self) -> None:
        from benchmarks.locomo.adapter import _ANSWER_SYSTEM

        assert "expert assistant" in _ANSWER_SYSTEM

    def test_legacy_adapter_has_answer_template(self) -> None:
        from benchmarks.locomo.adapter import _ANSWER_TEMPLATE

        assert "{context}" in _ANSWER_TEMPLATE
        assert "{question}" in _ANSWER_TEMPLATE
        assert "event_time" in _ANSWER_TEMPLATE

    def test_templates_are_consistent(self) -> None:
        from benchmarks.locomo.adapter import _ANSWER_TEMPLATE as legacy_tmpl
        from benchmarks.locomo.neo4j_adapter import _ANSWER_TEMPLATE as neo4j_tmpl

        assert legacy_tmpl == neo4j_tmpl

    def test_template_format_works(self) -> None:
        from benchmarks.locomo.neo4j_adapter import _ANSWER_TEMPLATE

        result = _ANSWER_TEMPLATE.format(
            context="[1] (event_time: 2023-05-08T13:56:00) Went to the vet",
            question="When did I go to the vet?",
        )
        assert "When did I go to the vet?" in result
        assert "event_time: 2023-05-08T13:56:00" in result


class TestNeo4jAdapterDefaults:
    """Verify default configuration values."""

    def test_default_top_k_is_10(self) -> None:
        import inspect

        from benchmarks.locomo.neo4j_adapter import Neo4jLoCoMoAdapter

        sig = inspect.signature(Neo4jLoCoMoAdapter.__init__)
        assert sig.parameters["top_k"].default == 10


class TestRunnerDefaults:
    """Verify runner CLI default values."""

    def test_runner_default_top_k(self) -> None:
        from benchmarks.locomo.runner import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.top_k == 10

    def test_runner_exclude_cat5_default_off(self) -> None:
        from benchmarks.locomo.runner import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.exclude_cat5 is False

    def test_runner_exclude_cat5_flag(self) -> None:
        from benchmarks.locomo.runner import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(["--exclude-cat5"])
        assert args.exclude_cat5 is True
