"""Translation orchestration, incremental reuse, validation, and freshness checks."""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import re
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from scripts.i18n import TRANSLATOR_VERSION
from scripts.i18n.engine import EngineClient, TranslationEngineError, TranslationResponse
from scripts.i18n.manifest import (
    load_manifest,
    save_manifest,
    strip_translation_header,
    translation_header,
)
from scripts.i18n.protect import ProtectedText, ProtectionError, protect_text
from scripts.i18n.segment import Segment, join_segments, split_markdown
from scripts.i18n.validate import ValidationError, has_translatable_japanese, validate_translation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "scripts" / "i18n" / "config.toml"
GLOSSARY_PATH = PROJECT_ROOT / "scripts" / "i18n" / "glossary.yaml"
MANIFEST_PATH = PROJECT_ROOT / "scripts" / "i18n" / "manifest.json"
_TRANSLATION_HEADER_LINE_RE = re.compile(
    r"\A<!-- 自動生成ファイル・編集禁止[^\n]* -->\n<!-- generator: [^\n]* -->\n(?:\n)?"
)
_BATCH_MARKER_RE = re.compile(r"⟦§(\d+)⟧")
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_INLINE_CODE_OR_COMMENT_RE = re.compile(r"(`+)[^`\n]*?\1|<!--.*?-->", re.DOTALL)
_MARKDOWN_LINK_DEST_RE = re.compile(
    r"(?P<prefix>!?\[[^\]\n]*\]\()(?P<destination><[^>\n]+>|[^\s)]+)"
    r"(?P<tail>(?:\s+(?:\"[^\"]*\"|'[^']*'))?\))"
)


@dataclass(frozen=True)
class Target:
    name: str
    src: str
    dst: str | dict[str, str]
    include: tuple[str, ...]
    copy: tuple[str, ...]
    header: str
    link_rewrite: tuple[tuple[str, str], ...] = ()
    langs: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative: str
    destination: Path


@dataclass
class Metrics:
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    validation_failures: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def add(self, response: TranslationResponse) -> None:
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.requests += 1

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at


@dataclass
class FileResult:
    changed: bool
    failed: int = 0


class PipelineError(RuntimeError):
    """Raised for invalid pipeline configuration or an unrecoverable input."""


def load_configuration() -> tuple[dict[str, Any], list[Target], list[dict[str, Any]]]:
    try:
        configuration = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        glossary_data = yaml.safe_load(GLOSSARY_PATH.read_text(encoding="utf-8")) or []
    except (OSError, tomllib.TOMLDecodeError, yaml.YAMLError) as exc:
        raise PipelineError(f"Unable to load i18n configuration: {exc}") from exc
    if not isinstance(glossary_data, list):
        raise PipelineError("The glossary must be a YAML list")
    for index, term in enumerate(glossary_data):
        if not isinstance(term, dict) or not all(key in term for key in ("ja", "en", "ko")):
            raise PipelineError(f"Invalid glossary entry at index {index}")
    targets = [
        Target(
            name=item["name"],
            src=item["src"],
            dst=item["dst"],
            include=tuple(item.get("include", ("**/*.md",))),
            copy=tuple(item.get("copy", ())),
            header=item.get("header", "none"),
            link_rewrite=tuple(tuple(pair) for pair in item.get("link_rewrite", ())),
            langs=tuple(item["langs"]) if item.get("langs") else None,
        )
        for item in configuration.get("targets", [])
    ]
    return configuration, targets, glossary_data


def _matches(path: str, patterns: tuple[str, ...] | list[str]) -> bool:
    posix = PurePosixPath(path)
    return any(
        fnmatch.fnmatchcase(path, pattern)
        or posix.match(pattern)
        or (pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:]))
        for pattern in patterns
    )


def _source_files(target: Target, lang: str, root: Path) -> list[SourceFile]:
    source = root / target.src
    if isinstance(target.dst, dict):
        destination_name = target.dst.get(lang)
        if not destination_name:
            return []
        if source.is_file():
            return [SourceFile(source, source.name, root / destination_name)]
        return []

    if not source.is_dir():
        return []
    patterns = (*target.include, *target.copy)
    output: list[SourceFile] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if _matches(relative, patterns):
            destination = root / target.dst.format(lang=lang) / Path(relative)
            output.append(SourceFile(path, relative, destination))
    return output


def _filter_files(files: list[SourceFile], only: str | None) -> list[SourceFile]:
    if not only:
        return files
    return [item for item in files if _matches(item.relative, (only,))]


def _target_for_command(targets: list[Target], command: str) -> list[Target]:
    return targets if command == "all" else [target for target in targets if target.name == command]


def _langs_for_target(target: Target, defaults: dict[str, Any], requested_langs: list[str] | None) -> list[str]:
    if isinstance(target.dst, dict):
        available = list(target.dst)
    else:
        available = list(target.langs or defaults.get("langs", ["en", "ko"]))
    return [lang for lang in (requested_langs or available) if lang in available]


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_boundaries(translated: str, source: str) -> str:
    """Restore the source section's leading/trailing newline counts.

    Models frequently compress the blank (separator) line between batch
    sections, which glued adjacent headings to the previous paragraph. Rebuild
    the boundary exactly so every heading stays at line start and the blank-line
    count between sections matches the original.
    """
    source_lead = len(source) - len(source.lstrip("\n"))
    source_trail = len(source) - len(source.rstrip("\n"))
    body = translated.strip("\n")
    return "\n" * source_lead + body + "\n" * source_trail


def _strip_source_headers(text: str) -> str:
    text, _ = strip_translation_header(text)
    return _TRANSLATION_HEADER_LINE_RE.sub("", text, count=1)


def _relevant_glossary(segment_text: str, glossary: list[dict[str, Any]], lang: str) -> list[dict[str, Any]]:
    return [term for term in glossary if str(term["ja"]) in segment_text]


def _section_hash(segment: Segment, glossary: list[dict[str, Any]]) -> str:
    related = _relevant_glossary(segment.text, glossary, "en")
    relevant_glossary = [
        {key: term.get(key) for key in ("ja", "en", "ko", "keep", "note") if key in term} for term in related
    ]
    value = {
        "text": segment.text.replace("\r\n", "\n").replace("\r", "\n").rstrip(),
        "glossary": relevant_glossary,
        "translator": TRANSLATOR_VERSION,
    }
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _glossary_prompt(terms: list[dict[str, Any]], lang: str) -> str:
    if not terms:
        return ""
    lines = []
    for term in terms:
        if term.get("keep"):
            lines.append(f"- Preserve exactly: {term['ja']}")
        else:
            lines.append(f"- {term['ja']} → {term.get(lang, term['en'])}")
    return "\n".join(lines)


def _system_prompt(lang: str, glossary: list[dict[str, Any]]) -> str:
    target_language = {"en": "English", "ko": "Korean", "zh": "Simplified Chinese"}.get(lang, lang)
    language_rule = {
        "en": "Write natural English; do not leave Japanese prose untranslated.",
        "ko": "Write natural Korean in Hangul; do not return Japanese prose or kana.",
        "zh": "Write natural Simplified Chinese; do not leave Japanese prose untranslated.",
    }.get(lang, "Translate all natural-language prose into the requested target language.")
    glossary_text = _glossary_prompt(glossary, lang)
    return (
        f"Translate the Japanese content into {target_language}. {language_rule} "
        "Return only the translation, without explanations. "
        "Preserve Markdown structure, heading levels, table dimensions, links, identifiers, and every sentinel exactly "
        "once and in its original order. Preserve each ⟦§number⟧ section marker exactly once and in order. "
        "Never translate text inside sentinels. Do not add or remove section markers. "
        "In YAML frontmatter, translate only the exposed values; preserve all keys and formatting. Keep the literal prefix "
        "'Use when:' unchanged."
        + (f"\n\nGlossary for terms present in this section:\n{glossary_text}" if glossary_text else "")
    )


def _response_text(response: Any) -> TranslationResponse:
    if isinstance(response, TranslationResponse):
        return response
    if isinstance(response, str):
        return TranslationResponse(response)
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return TranslationResponse(
            text,
            int(getattr(response, "input_tokens", 0)),
            int(getattr(response, "output_tokens", 0)),
            str(getattr(response, "model", "")),
        )
    raise TranslationEngineError("Translation engine returned an unsupported response object")


def _rewrite_markdown_links(text: str, rewrites: tuple[tuple[str, str], ...], *, lang: str, readme: str) -> str:
    """Rewrite configured prefixes in Markdown link destinations, never in code or prose."""
    if not rewrites:
        return text

    hidden: list[str] = []

    def stash(value: str) -> str:
        marker = f"\x00I18N_LINK_MASK_{len(hidden)}\x00"
        hidden.append(value)
        return marker

    lines = text.splitlines(keepends=True)
    visible: list[str] = []
    index = 0
    while index < len(lines):
        opening = _FENCE_OPEN_RE.match(lines[index])
        if not opening:
            visible.append(lines[index])
            index += 1
            continue

        fence = opening.group(1)
        fence_char, fence_size = fence[0], len(fence)
        block = [lines[index]]
        index += 1
        while index < len(lines):
            line = lines[index]
            block.append(line)
            index += 1
            closing = re.match(r"^ {0,3}(`+|~+)[ \t]*(?:\r?\n)?$", line)
            if closing and closing.group(1)[0] == fence_char and len(closing.group(1)) >= fence_size:
                break
        visible.append(stash("".join(block)))

    text = _INLINE_CODE_OR_COMMENT_RE.sub(lambda match: stash(match.group(0)), "".join(visible))
    context = {"lang": lang, "readme": readme}

    def rewrite(match: re.Match[str]) -> str:
        destination = match.group("destination")
        wrapped = destination.startswith("<") and destination.endswith(">")
        value = destination[1:-1] if wrapped else destination
        for source_prefix, target_prefix in rewrites:
            value = value.replace(source_prefix, target_prefix.format(**context))
        if wrapped:
            value = f"<{value}>"
        return f"{match.group('prefix')}{value}{match.group('tail')}"

    text = _MARKDOWN_LINK_DEST_RE.sub(rewrite, text)
    for marker_index, value in enumerate(hidden):
        text = text.replace(f"\x00I18N_LINK_MASK_{marker_index}\x00", value)
    return text


def _parse_batch_response(text: str, expected_indices: list[int]) -> dict[int, str] | None:
    matches = list(_BATCH_MARKER_RE.finditer(text))
    found = [int(match.group(1)) for match in matches]
    if found != expected_indices:
        return None
    if text[: matches[0].start()].strip():
        return None
    sections: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.end() : end]
        if section.startswith("\r\n"):
            section = section[2:]
        elif section.startswith("\n"):
            section = section[1:]
        if index + 1 < len(matches):
            if section.endswith("\r\n"):
                section = section[:-2]
            elif section.endswith("\n"):
                section = section[:-1]
        sections[expected_indices[index]] = section
    if any(not section.strip() for section in sections.values()):
        return None
    return sections


class Translator:
    def __init__(
        self,
        *,
        root: Path = PROJECT_ROOT,
        configuration: dict[str, Any],
        targets: list[Target],
        glossary: list[dict[str, Any]],
        engines: dict[str, Any] | None = None,
        manifest_path: Path = MANIFEST_PATH,
    ) -> None:
        self.root = root
        self.configuration = configuration
        self.targets = targets
        self.glossary = glossary
        self.manifest_path = manifest_path
        self.manifest = load_manifest(manifest_path)
        definitions = configuration.get("engines", {})
        self.engines = engines or {name: EngineClient(name, definition) for name, definition in definitions.items()}
        self.metrics = Metrics()
        self._metrics_lock = threading.Lock()
        self.warnings: list[str] = []
        self.models: dict[str, str] = {}

    def _record_validation_failure(self) -> None:
        with self._metrics_lock:
            self.metrics.validation_failures += 1

    def _call(
        self, engine_name: str, lang: str, texts: list[tuple[int, ProtectedText]], batch: bool
    ) -> dict[int, str] | None:
        engine = self.engines.get(engine_name)
        if engine is None:
            return None
        if batch:
            glossary = []
            for _, protected in texts:
                visible = re.sub(r"⟦P\d+⟧", " ", protected.text)
                for term in _relevant_glossary(visible, self.glossary, lang):
                    if term not in glossary:
                        glossary.append(term)
            body = "\n".join(f"⟦§{index}⟧\n{protected.text}" for index, protected in texts)
            system_prompt = _system_prompt(lang, glossary)
            user_prompt = body
        else:
            index, protected = texts[0]
            glossary = _relevant_glossary(re.sub(r"⟦P\d+⟧", " ", protected.text), self.glossary, lang)
            system_prompt = _system_prompt(lang, glossary)
            user_prompt = protected.text

        with self._metrics_lock:
            self.metrics.requests += 1
        response = _response_text(engine.translate(system_prompt, user_prompt))
        with self._metrics_lock:
            self.metrics.input_tokens += response.input_tokens
            self.metrics.output_tokens += response.output_tokens
            if response.model:
                self.models[engine_name] = response.model
        if batch:
            parsed = _parse_batch_response(response.text, [index for index, _ in texts])
            if parsed is None:
                self._record_validation_failure()
            return parsed
        return {texts[0][0]: response.text}

    def _translate_pending(
        self,
        pending: list[tuple[int, ProtectedText, Segment]],
        *,
        lang: str,
        engine_name: str,
        fallback_engine: str,
        existing_segments: dict[int, Segment],
    ) -> tuple[dict[int, str], dict[int, str]]:
        if not pending:
            return {}, {}
        max_chars = int(self.configuration.get("defaults", {}).get("batch_chars", 6000))
        batches: list[list[tuple[int, ProtectedText, Segment]]] = []
        current: list[tuple[int, ProtectedText, Segment]] = []
        current_chars = 0
        for item in pending:
            length = len(item[1].text)
            if current and current_chars + length > max_chars:
                batches.append(current)
                current = []
                current_chars = 0
            current.append(item)
            current_chars += length
        if current:
            batches.append(current)

        translated: dict[int, str] = {}
        statuses: dict[int, str] = {}
        responses: dict[int, str] = {}
        initial_attempted: set[int] = set()
        with ThreadPoolExecutor(
            max_workers=max(1, int(self.configuration.get("defaults", {}).get("concurrency", 4)))
        ) as pool:
            future_map = {
                pool.submit(
                    self._call, engine_name, lang, [(index, protected) for index, protected, _ in batch], len(batch) > 1
                ): batch
                for batch in batches
            }
            for future in as_completed(future_map):
                batch = future_map[future]
                initial_attempted.update(index for index, _, _ in batch)
                try:
                    result = future.result()
                except Exception as exc:  # Errors are sanitized by EngineClient; test engines may raise.
                    self.warnings.append(f"Translation request failed: {type(exc).__name__}")
                    result = None
                if result is None:
                    continue
                responses.update(result)

        by_index = {index: (protected, source) for index, protected, source in pending}
        for index, initial_text in responses.items():
            protected, source = by_index[index]
            try:
                if not initial_text.strip():
                    raise ValidationError("Engine returned an empty section")
                translated_text = protected.restore(initial_text)
                validate_translation(
                    source.text,
                    translated_text,
                    lang,
                    source_visible=protected.text,
                    translated_visible=initial_text,
                    frontmatter=source.kind == "frontmatter",
                )
                translated[index] = translated_text
                statuses[index] = "translated"
            except (ProtectionError, ValidationError) as exc:
                self._record_validation_failure()
                self.warnings.append(f"Section {index} failed validation: {exc}")

        missing_or_invalid = [item for item in pending if item[0] not in translated]
        for index, protected, source in missing_or_invalid:
            candidate: str | None = None
            attempts = 1 if index in initial_attempted else 2
            for attempt in range(attempts):
                try:
                    result = self._call(engine_name, lang, [(index, protected)], False)
                    candidate = result[index] if result else None
                    restored = protected.restore(candidate) if candidate is not None and candidate.strip() else None
                    if restored is None:
                        raise ValidationError("Engine returned an empty response")
                    validate_translation(
                        source.text,
                        restored,
                        lang,
                        source_visible=protected.text,
                        translated_visible=candidate,
                        frontmatter=source.kind == "frontmatter",
                    )
                    translated[index] = restored
                    statuses[index] = "translated"
                    break
                except Exception as exc:
                    self._record_validation_failure()
                    self.warnings.append(f"Section {index} retry {attempt + 1} failed: {type(exc).__name__}")
            if index in translated:
                continue

            if fallback_engine:
                try:
                    result = self._call(fallback_engine, lang, [(index, protected)], False)
                    candidate = result[index] if result else None
                    restored = protected.restore(candidate) if candidate is not None and candidate.strip() else None
                    if restored is None:
                        raise ValidationError("Fallback engine returned an empty response")
                    validate_translation(
                        source.text,
                        restored,
                        lang,
                        source_visible=protected.text,
                        translated_visible=candidate,
                        frontmatter=source.kind == "frontmatter",
                    )
                    translated[index] = restored
                    statuses[index] = "translated"
                    continue
                except Exception as exc:
                    self._record_validation_failure()
                    self.warnings.append(f"Section {index} fallback failed: {type(exc).__name__}")

            if index in existing_segments:
                translated[index] = existing_segments[index].text
            else:
                translated[index] = source.text
            statuses[index] = "failed"
        return translated, statuses

    def translate_file(
        self,
        target: Target,
        item: SourceFile,
        lang: str,
        *,
        force: bool = False,
        engine_name: str,
        fallback_engine: str,
    ) -> FileResult:
        source_bytes = item.path.read_bytes()
        digest = hashlib.sha256(source_bytes).hexdigest()
        record_key = item.destination.relative_to(self.root).as_posix()
        existing_record = self.manifest["files"].get(record_key)

        if any(_matches(item.relative, (pattern,)) for pattern in target.copy):
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            item.destination.write_bytes(source_bytes)
            self.manifest["files"][record_key] = {
                "source": item.path.relative_to(self.root).as_posix(),
                "source_sha256": digest,
                "section_hashes": [],
                "statuses": [],
                "generated": datetime.now(UTC).date().isoformat(),
                "engine": "copy",
                "model": "copy",
                "translator_version": TRANSLATOR_VERSION,
            }
            return FileResult(changed=True)

        source_text = _strip_source_headers(source_bytes.decode("utf-8"))
        source_segments = split_markdown(source_text)
        hashes = [_section_hash(segment, self.glossary) for segment in source_segments]
        existing_text = ""
        output_segments: list[Segment] = []
        if item.destination.exists():
            existing_text = item.destination.read_text(encoding="utf-8")
            existing_text, _header_hash = strip_translation_header(existing_text)
            output_segments = split_markdown(existing_text)

        reuse: dict[int, int] = {}
        fallback_matches: dict[int, int] = {}
        if existing_record and item.destination.exists():
            old_hashes = existing_record.get("section_hashes", [])
            if force and len(output_segments) == len(source_segments):
                fallback_matches = dict(enumerate(range(len(output_segments))))
            elif not force and len(old_hashes) == len(output_segments):
                matcher = difflib.SequenceMatcher(a=old_hashes, b=hashes, autojunk=False)
                for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
                    if tag == "equal":
                        matches = {
                            new_index: old_index
                            for old_index, new_index in zip(
                                range(old_start, old_end), range(new_start, new_end), strict=True
                            )
                        }
                        reuse.update(matches)
                        fallback_matches.update(matches)
                    elif tag == "replace" and old_end - old_start == new_end - new_start:
                        fallback_matches.update(
                            {
                                new_index: old_index
                                for old_index, new_index in zip(
                                    range(old_start, old_end), range(new_start, new_end), strict=True
                                )
                            }
                        )
            elif existing_record and len(old_hashes) != len(output_segments):
                warning = f"{record_key}: output section count differs from manifest; retranslating all sections"
                self.warnings.append(warning)
                print(f"WARNING: {warning}", file=sys.stderr)

        pending: list[tuple[int, ProtectedText, Segment]] = []
        translated_by_index: dict[int, str] = {}
        statuses_by_index: dict[int, str] = {}
        for index, segment in enumerate(source_segments):
            old_index = reuse.get(index)
            if old_index is not None:
                translated_by_index[index] = output_segments[old_index].text
                old_statuses = existing_record.get("statuses", []) if existing_record else []
                old_status = old_statuses[old_index] if old_index < len(old_statuses) else "translated"
                if old_status != "failed":
                    statuses_by_index[index] = old_status
                    continue
            if not has_translatable_japanese(segment.text):
                # No translatable prose: keep the original text without calling
                # the model (avoids empty prompts and preamble answers).
                translated_by_index[index] = segment.text
                statuses_by_index[index] = "translated"
                continue
            protected = protect_text(
                segment.text, frontmatter=segment.kind == "frontmatter", sentinel_start=index * 10000
            )
            pending.append((index, protected, segment))

        existing_by_source_index = {
            new_index: output_segments[old_index] for new_index, old_index in fallback_matches.items()
        }
        new_translations, new_statuses = self._translate_pending(
            pending,
            lang=lang,
            engine_name=engine_name,
            fallback_engine=fallback_engine,
            existing_segments=existing_by_source_index,
        )
        translated_by_index.update(new_translations)
        statuses_by_index.update(new_statuses)

        translated_segments = []
        for index in range(len(source_segments)):
            text = translated_by_index[index]
            source = source_segments[index].text
            translated_segments.append(Segment(_normalize_boundaries(text, source), source_segments[index].kind))
        translated_text = join_segments(translated_segments)
        destination_name = Path(target.dst[lang]).name if isinstance(target.dst, dict) else ""
        translated_text = _rewrite_markdown_links(
            translated_text,
            target.link_rewrite,
            lang=lang,
            readme=destination_name,
        )

        failed = sum(status == "failed" for status in statuses_by_index.values())
        generated = datetime.now(UTC).date().isoformat()
        engine_definition = self.configuration.get("engines", {}).get(engine_name, {})
        model_name = self.models.get(engine_name) or str(
            engine_definition.get("model", engine_definition.get("deployment", engine_name))
        )
        if target.header == "html_comment":
            header = translation_header(
                source_path=item.path.relative_to(self.root).as_posix(),
                source_sha256=digest,
                generated=generated,
                engine=engine_name,
                model=model_name,
                translator=TRANSLATOR_VERSION,
            )
            translated_text = header + translated_text

        item.destination.parent.mkdir(parents=True, exist_ok=True)
        item.destination.write_text(translated_text, encoding="utf-8")
        self.manifest["files"][record_key] = {
            "source": item.path.relative_to(self.root).as_posix(),
            "source_sha256": digest,
            "section_hashes": hashes,
            "statuses": [statuses_by_index.get(index, "translated") for index in range(len(hashes))],
            "generated": generated,
            "engine": engine_name,
            "model": model_name,
            "translator_version": TRANSLATOR_VERSION,
        }
        return FileResult(changed=True, failed=failed)

    def check(self, targets: list[Target], langs: dict[str, list[str]], only: str | None = None) -> list[str]:
        issues: list[str] = []
        expected_keys: set[str] = set()
        scoped_targets: list[Target] = []
        for target in targets:
            scoped_targets.append(target)
            for lang in langs[target.name]:
                files = _filter_files(_source_files(target, lang, self.root), only)
                for item in files:
                    key = item.destination.relative_to(self.root).as_posix()
                    expected_keys.add(key)
                    record = self.manifest["files"].get(key)
                    if not item.destination.exists():
                        issues.append(f"Missing translation: {key}")
                        continue
                    if not isinstance(record, dict):
                        issues.append(f"Generated file is not in manifest: {key}")
                        continue
                    section_hashes = record.get("section_hashes", [])
                    statuses = record.get("statuses", [])
                    if not isinstance(section_hashes, list) or not isinstance(statuses, list):
                        issues.append(f"Invalid section manifest: {key}")
                        continue
                    digest = _source_hash(item.path)
                    source_key = item.path.relative_to(self.root).as_posix()
                    if record.get("source") != source_key or record.get("source_sha256") != digest:
                        issues.append(f"Source changed: {key}")
                    if any(_matches(item.relative, (pattern,)) for pattern in target.copy):
                        if item.destination.read_bytes() != item.path.read_bytes():
                            issues.append(f"Copied file differs from source: {key}")
                        if section_hashes or statuses:
                            issues.append(f"Invalid copy manifest entry: {key}")
                        continue
                    body = item.destination.read_text(encoding="utf-8")
                    _, header_hash = strip_translation_header(body)
                    if target.header == "html_comment" and header_hash != digest:
                        issues.append(f"Translation header is missing or stale: {key}")
                    if target.header == "none" and body.startswith("<!-- 自動翻訳"):
                        issues.append(f"Unexpected translation header: {key}")
                    if len(statuses) != len(section_hashes):
                        issues.append(f"Invalid section manifest: {key}")
                    if any(status == "failed" for status in statuses):
                        issues.append(f"Failed translation section: {key}")

        if only is None:
            for target in scoped_targets:
                for lang in langs[target.name]:
                    if isinstance(target.dst, dict):
                        destination_roots = [self.root / target.dst[lang]]
                    else:
                        destination_roots = [self.root / target.dst.format(lang=lang)]
                    for destination_root in destination_roots:
                        if destination_root.is_file():
                            output_files = [destination_root]
                        elif destination_root.is_dir():
                            output_files = [path for path in destination_root.rglob("*") if path.is_file()]
                        else:
                            output_files = []
                        for output in output_files:
                            key = output.relative_to(self.root).as_posix()
                            if isinstance(target.dst, dict):
                                if key not in expected_keys:
                                    issues.append(f"Generated file has no source in current target: {key}")
                                continue
                            relative = output.relative_to(destination_root).as_posix()
                            if _matches(relative, (*target.include, *target.copy)) and key not in expected_keys:
                                issues.append(f"Generated file has no source in current target: {key}")

            selected_prefixes = {
                target.dst.format(lang=lang) + "/"
                for target in targets
                if target.name in langs and isinstance(target.dst, str)
                for lang in langs[target.name]
            }
            selected_readmes = {
                target.dst[lang]
                for target in targets
                if target.name in langs and isinstance(target.dst, dict)
                for lang in langs[target.name]
            }
            for key, record in self.manifest["files"].items():
                if not isinstance(record, dict):
                    issues.append(f"Invalid manifest record: {key}")
                    continue
                in_scope = key.startswith(tuple(selected_prefixes)) or key in selected_readmes
                if in_scope and key not in expected_keys:
                    issues.append(f"Manifest entry is outside the selected source set: {key}")
        return issues

    def run(
        self,
        command: str,
        *,
        requested_langs: list[str] | None = None,
        only: str | None = None,
        force: str | None = None,
        dry_run: bool = False,
        check: bool = False,
        engine_name: str | None = None,
        stats: bool = False,
    ) -> int:
        defaults = self.configuration.get("defaults", {})
        selected_targets = _target_for_command(self.targets, command)
        selected_langs = {
            target.name: _langs_for_target(target, defaults, requested_langs) for target in selected_targets
        }
        if check:
            issues = self.check(selected_targets, selected_langs, only=only)
            for issue in issues:
                print(f"ERROR: {issue}", file=sys.stderr)
            if issues:
                print(
                    "Ask the PdM to run scripts/translate.py after merging Japanese source updates; "
                    "CI does not translate files.",
                    file=sys.stderr,
                )
                return 1
            print("i18n freshness check passed")
            return 0

        active_engine = engine_name or str(defaults.get("engine", "local"))
        fallback_engine = str(defaults.get("fallback_engine", ""))
        failures = 0
        for target in selected_targets:
            for lang in selected_langs[target.name]:
                for item in _filter_files(_source_files(target, lang, self.root), only):
                    forced = bool(force and _matches(item.relative, (force,)))
                    if dry_run:
                        print(
                            f"Would translate {item.path.relative_to(self.root)} -> {item.destination.relative_to(self.root)}"
                        )
                        continue
                    print(
                        f"Processing {item.path.relative_to(self.root)} -> {item.destination.relative_to(self.root)}",
                        flush=True,
                    )
                    try:
                        result = self.translate_file(
                            target,
                            item,
                            lang,
                            force=forced,
                            engine_name=active_engine,
                            fallback_engine=fallback_engine,
                        )
                    except Exception as exc:
                        failures += 1
                        self.warnings.append(f"{item.path.relative_to(self.root)}: {type(exc).__name__}: {exc}")
                        print(f"ERROR: {self.warnings[-1]}", file=sys.stderr)
                        continue
                    failures += result.failed
        if not dry_run:
            save_manifest(self.manifest_path, self.manifest)
        for warning in self.warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        if stats:
            print(
                "Stats: "
                f"input_tokens={self.metrics.input_tokens} "
                f"output_tokens={self.metrics.output_tokens} "
                f"elapsed_seconds={self.metrics.elapsed_seconds:.2f} "
                f"requests={self.metrics.requests} "
                f"validation_failures={self.metrics.validation_failures}"
            )
        if failures:
            print(f"Translation completed with {failures} failed section(s).", file=sys.stderr)
            return 2
        return 0


def freshness_check(command: str, *, only: str | None = None, requested_langs: list[str] | None = None) -> int:
    configuration, targets, glossary = load_configuration()
    translator = Translator(configuration=configuration, targets=targets, glossary=glossary)
    return translator.run(command, only=only, requested_langs=requested_langs, check=True)
