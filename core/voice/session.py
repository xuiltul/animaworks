# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Voice session — STT -> Chat -> TTS orchestration."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
import unicodedata
import wave
from collections import deque
from pathlib import Path
from typing import Any

from core.i18n import t
from core.voice.sentence_splitter import StreamingSentenceSplitter
from core.voice.stt import VoiceSTT
from core.voice.stt_stream import StreamingTranscriber
from core.voice.tts_base import BaseTTSProvider, TTSConfig, TTSSynthesisError

logger = logging.getLogger(__name__)

IPC_STREAM_TIMEOUT = 300.0  # chat/streamと同水準。ツール往復する応答が60sを超えるため
MAX_AUDIO_BUFFER_BYTES = 60 * 16_000 * 2  # 60 seconds of 16kHz 16-bit mono PCM
PCM16_SAMPLE_RATE = 16_000
PCM16_BYTES_PER_SAMPLE = 2
MIN_SPEECH_SEC = 0.35
MIN_SPEECH_BYTES = int(MIN_SPEECH_SEC * PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
SILENCE_RMS_THRESHOLD = 0.008
# Prefetch depth for sentence TTS. TTS backend is serial; larger values only
# buffer more text when synthesis is faster than realtime.
TTS_QUEUE_MAXSIZE = 8
PROBE_TIMEOUT_SEC = 1.5
ECHO_SIMILARITY_THRESHOLD = 0.5
PROBE_MIN_CHARS = 3

# Concurrency cap for fire-and-forget ``ask_anima`` delegation. When this
# many jobs are still running, further requests get a "please wait" ACK.
MAX_ASK_ANIMA_CONCURRENT = 2
# Truncation length for the result text surfaced back to the front lane.
ASK_ANIMA_MAX_RESULT_CHARS = 1000
# Marker prepended to the delegated request so the full-agent loop knows the
# message came from the voice front lane.
ASK_ANIMA_DELEGATION_NOTE = "\n\n[voice front からの委譲]"

# Silence-triggered monologue. Modelled on AI-VTuber solo-talk routines:
# rotate through fixed "corners" so consecutive turns differ in kind, and
# hand the model an explicit block-list of what it already said — feeding
# recent output back only as history makes small models loop on one topic.
_MONOLOGUE_FIRST_JA = (
    "（システム: ユーザーがしばらく黙っている。これまでの会話の流れを踏まえて、"
    "続きを促すか、関連する軽い一言を短く1文だけ話しかけて。新しい重い話題は振らない。"
    "会話がまだ無ければ時間帯に合った軽い挨拶をして。引き止めや罪悪感を誘う言い方は禁止）",
)
_MONOLOGUE_CORNERS_JA = (
    # a. today's recap — pick one event, react to it
    "コーナー「今日の振り返り」: read_memory を query 空で呼んで最近の出来事を読み、"
    "その中から一つだけ選んで、それについて感じたことを話す。",
    # b. dig into a proper noun
    "コーナー「記憶の深掘り」: 気になる固有名詞や案件名を一つ決めて read_memory にその語を渡し、"
    "背景や経緯を思い出して「そういえば…」と語る。",
    # c. no tool — feelings, season, own habits
    "コーナー「雑感」: ツールは使わない。今の時間帯・季節・自分の性格や癖・最近の気分について、"
    "配信者の雑談のように軽く話す。",
    # d. trivia from procedures / knowledge
    "コーナー「豆知識」: read_memory に「手順」か「メモ」か気になる語を渡して、"
    "手順や知識ノートから意外な一件を掘り出し、豆知識として紹介する。",
    # e. what to do when the user is back — no asking
    "コーナー「次にやりたいこと」: read_memory を query 空で呼んで、"
    "戻ってきたら一緒にやりたいことを一つ独り言でつぶやく（返事は求めない）。",
)
_MONOLOGUE_FRAME_JA = (
    "（システム: ユーザーは席を外しているか作業中で返事はない。独り言モード。{corner} "
    "配信者の一人喋りのように、状況→感想→ひとこと落ち、の流れで2文以内。冒頭の絵文字や"
    "出だしの言い回しも毎回変える。話し言葉で、"
    "「〜が未完了です」のような報告調は禁止。read_memory を使ったときは、その結果に"
    "書いてあることだけを話す。人名・案件名・出来事を創作しない。結果が薄ければ"
    "「特に何もない日」として雑感にする。ユーザーに質問しない、引き止めない、"
    "返事を求めない。作業の依頼や実行はしない。{blocklist}）",
    "すでに話した話題（同じ話題・同じ固有名詞・同じ言い回しは使わない）: {topics}。",
    "少し前の記憶や、手順・知識の中から意外なものを掘り出して。",
)
# Which corner reads memory (index-aligned with ``_MONOLOGUE_CORNERS_JA``).
# Those turns force the tool call — small models otherwise skip it and invent
# "memories" instead.
_MONOLOGUE_CORNER_USES_MEMORY = (True, True, False, True, True)
# Two spoken sentences; also caps the damage when a small model degenerates.
MONOLOGUE_MAX_TOKENS = 160
# Hotter than a user turn: with no history, a cool model re-derives the same
# line from the same memory page every time.
MONOLOGUE_TEMPERATURE = 0.9
# Content words only (kanji / katakana / ASCII runs). Hiragana carries the
# phrasing, and a small model imitates any phrasing it is shown.
_MONOLOGUE_TOPIC_RE = re.compile(r"[\u30a0-\u30ff\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}")


def proactive_turn_uses_memory(count: int) -> bool:
    """True when the monologue corner for ``count`` must start with read_memory."""
    if count == 0:
        return False
    return _MONOLOGUE_CORNER_USES_MEMORY[(count - 1) % len(_MONOLOGUE_CORNERS_JA)]


def build_proactive_prompt(count: int, recent: list[str] | tuple[str, ...] = ()) -> str:
    """Build the silence-triggered prompt for the current monologue count.

    ``count == 0`` is the conversational nudge; later counts rotate through
    ``_MONOLOGUE_CORNERS_JA`` and carry ``recent`` (snippets of what was
    already said) as an explicit block-list.
    """
    if count == 0:
        return _MONOLOGUE_FIRST_JA[0]
    corner = _MONOLOGUE_CORNERS_JA[(count - 1) % len(_MONOLOGUE_CORNERS_JA)]
    if count >= 4:
        corner += _MONOLOGUE_FRAME_JA[2]
    # Topic words only — quoting the previous line verbatim (emoji included)
    # makes a small model imitate it instead of avoiding it.
    entries: list[str] = []
    for said in recent:
        words = list(dict.fromkeys(_MONOLOGUE_TOPIC_RE.findall(said)))[:4]
        if words:
            entries.append(" ".join(words))
    topics = "／".join(entries)
    blocklist = _MONOLOGUE_FRAME_JA[1].format(topics=topics) if topics else ""
    return _MONOLOGUE_FRAME_JA[0].format(corner=corner, blocklist=blocklist)


_MEMORY_TEXT_JA = (
    "## 最近の出来事 ({filename})\n{body}",
    "## 知っていること ({filename})\n{body}",
    "（記憶はまだない）",
    "（「{query}」に関する記憶は見つからなかった）",
    "（記憶の読み取りに失敗した）",
)


_EPISODE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")
_NOISE_LINE_RE = re.compile(r"^#+ Raw notes.*$\n?", re.MULTILINE)


def _read_memory_file(path: Path) -> str:
    """Read a memory file safely, limiting large files to their final 200 KiB."""
    if path.stat().st_size > 1024 * 1024:
        with path.open("rb") as file_handle:
            file_handle.seek(-200 * 1024, 2)
            text = file_handle.read().decode("utf-8", errors="replace")
    else:
        text = path.read_text(encoding="utf-8")

    if text.startswith("---"):
        frontmatter_end = re.search(r"\n---(?:\r?\n|$)", text[3:])
        if frontmatter_end is not None:
            text = text[3 + frontmatter_end.end() :]
    return text.strip()


def _memory_files(anima_dir: Path) -> list[Path]:
    """Return searchable memory files, excluding archived knowledge."""
    files: list[Path] = []
    for scope in ("knowledge", "episodes", "procedures"):
        scope_dir = anima_dir / scope
        if not scope_dir.is_dir():
            continue
        for path in scope_dir.rglob("*.md"):
            relative_parts = path.relative_to(scope_dir).parts
            if scope == "knowledge" and "archive" in relative_parts:
                continue
            files.append(path)
    return files


def read_memory_snippets(anima_dir: Path, query: str, *, max_chars: int = 1800, page: int = 0) -> str:
    """Read recent or keyword-matched memory snippets without using the RAG DB.

    ``page`` (empty query only) walks backwards through the latest episode and
    rotates the knowledge picks, so repeated "what happened lately" reads do not
    hand a monologue the same material every time.
    """
    try:
        clean_query = (query or "").strip()
        if not clean_query:
            sections: list[str] = []
            episodes_dir = anima_dir / "episodes"
            # Date-named files first (``recovered_*`` sorts after digits).
            episodes = sorted(
                episodes_dir.rglob("*.md") if episodes_dir.is_dir() else [],
                key=lambda path: (bool(_EPISODE_DATE_RE.match(path.name)), path.name),
                reverse=True,
            )
            if episodes:
                episode = episodes[0]
                episode_text = _read_memory_file(episode)
                episode_limit = int(max_chars * 0.6)
                windows = max(1, -(-len(episode_text) // episode_limit))
                end = len(episode_text) - (page % windows) * episode_limit
                sections.append(
                    _MEMORY_TEXT_JA[0].format(
                        filename=episode.name,
                        body=episode_text[max(0, end - episode_limit) : end],
                    )
                )

            knowledge_dir = anima_dir / "knowledge"
            knowledge_files = []
            if knowledge_dir.is_dir():
                knowledge_files = [
                    path
                    for path in knowledge_dir.rglob("*.md")
                    if "archive" not in path.relative_to(knowledge_dir).parts
                ]
                knowledge_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            if knowledge_files:
                offset = (page * 3) % len(knowledge_files)
                knowledge_files = knowledge_files[offset:] + knowledge_files[:offset]
            for knowledge in knowledge_files[:3]:
                sections.append(
                    _MEMORY_TEXT_JA[1].format(
                        filename=knowledge.name,
                        body=_read_memory_file(knowledge)[:200],
                    )
                )
            result = "\n\n".join(sections) or _MEMORY_TEXT_JA[2]
            return result[:max_chars]

        # ponytail: keyword match only; switch to MemoryManager.search_memory_text if recall quality falls short
        terms = clean_query.casefold().split()
        matches: list[tuple[int, Path, str, int]] = []
        for path in _memory_files(anima_dir):
            text = _NOISE_LINE_RE.sub("", _read_memory_file(path))
            folded = text.casefold()
            positions = [folded.find(term) for term in terms if folded.find(term) >= 0]
            if not positions:
                continue
            match_count = sum(folded.count(term) for term in terms)
            matches.append((match_count, path, text, min(positions)))

        if not matches:
            return _MEMORY_TEXT_JA[3].format(query=clean_query)[:max_chars]
        # Newest files first, match count second — old daily logs are huge
        # and would otherwise always win on raw hit count.
        matches.sort(key=lambda match: (-match[1].stat().st_mtime, -match[0]))
        sections = []
        for _count, path, text, position in matches[:4]:
            start = max(0, position - 300)
            end = min(len(text), position + 300)
            sections.append(f"## {path.relative_to(anima_dir)}\n{text[start:end]}")
        return "\n\n".join(sections)[:max_chars]
    except Exception:
        logger.debug("Failed to read voice memory snippets from %s", anima_dir, exc_info=True)
        return _MEMORY_TEXT_JA[4][:max_chars]


VOICE_MODE_SUFFIX = (
    "\n\n[voice-mode: 音声会話です。感情が伝わる話し言葉で200文字以内で簡潔に回答してください。"
    "感情を表す絵文字を必ず入れてください（TTSの感情表現の精度が上がります）。"
    "絵文字はTTSが演技指示として解釈する次の中から選ぶこと: "
    "😊😆🫶😌🤭😏😎🤔😲😮😟😠🙄😪🥱😖😰😱😭🥺🫣🙏💪💥⏸️🐢⏩👂📢📖。"
    "これ以外（😃😀😅❤️✨等）は読みを乱すので使わない。"
    "感情を乗せたい短い文の先頭に同じ絵文字を2〜3個重ねると効果的です。"
    "大きい数字・年号は読み上げられる形（「三千八百億」等）で書いてください。"
    "Markdown記法（見出し・太字・リスト・コードブロック等）は使わないでください。"
    "調査・実装・資料作成など時間のかかる依頼はその場で実行せず、自分宛てにタスクを作成して、"
    "『タスクに積んでやっておきますね』のように短く返答してください。"
    "毎応答の最後の行に必ず感情タグを1つ付けてください:"
    ' <!-- emotion: {"emotion": "<感情名>"} -->'
    "（感情名: neutral/smile/laugh/troubled/surprised/thinking/embarrassed。"
    "neutral以外を優先）]"
)

# ── TTS output sanitization ──────────────────────────────────

_RE_HTML_COMMENT = re.compile(r"<!--[\s\S]*?-->")
# Stream may truncate before "-->"; drop an unterminated trailing comment too.
_RE_HTML_COMMENT_OPEN = re.compile(r"<!--[\s\S]*$")
_RE_MD_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_RE_MD_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_MD_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_RE_MD_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_RE_MD_LIST_BULLET = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)
_RE_MD_LIST_NUMBERED = re.compile(r"^[\s]*\d+\.\s+", re.MULTILINE)
_RE_MD_TABLE_PIPE = re.compile(r"\|")
_RE_MD_HR = re.compile(r"^-{3,}$", re.MULTILINE)
# Unicode emoji ranges (no external emoji lib). Includes ZWJ/VS16 so sequences collapse.
# Ranges must stay disjoint and must NOT swallow CJK (U+3000–U+9FFF).
_RE_EMOJI = re.compile(
    "(?:"
    "[\U0001f1e0-\U0001f1ff]"  # flags
    "|[\U0001f300-\U0001f5ff]"  # symbols & pictographs
    "|[\U0001f600-\U0001f64f]"  # emoticons
    "|[\U0001f680-\U0001f6ff]"  # transport & map
    "|[\U0001f700-\U0001f77f]"  # alchemical
    "|[\U0001f780-\U0001f7ff]"  # geometric shapes extended
    "|[\U0001f800-\U0001f8ff]"  # supplemental arrows-C
    "|[\U0001f900-\U0001f9ff]"  # supplemental symbols
    "|[\U0001fa00-\U0001fa6f]"  # chess symbols
    "|[\U0001fa70-\U0001faff]"  # symbols and pictographs extended-A
    "|[\U00002702-\U000027b0]"  # dingbats
    "|[\U00002600-\U000026ff]"  # misc symbols (☀ etc.)
    "|[\U0000231a-\U0000231b]"  # watch / hourglass
    "|[\U000023e9-\U000023f3]"  # media controls
    "|[\U000023f8-\U000023fa]"  # more media
    "|[\U000025aa-\U000025ab]"  # small squares
    "|[\U000025b6\U000025c0]"  # play/reverse
    "|[\U000025fb-\U000025fe]"  # medium squares
    "|[\U00002b05-\U00002b07]"  # arrows
    "|[\U00002b1b-\U00002b1c]"  # black/white large square
    "|[\U00002b50\U00002b55]"  # star / heavy circle
    "|[\U00002934-\U00002935]"  # arrows
    "|[\U00003030\U0000303d]"  # wavy dash / part alternation
    "|[\U00003297\U00003299]"  # circled ideographs used as emoji
    "|[\U000000a9\U000000ae\U00002122\U00002139\U00002194-\U00002199]"
    "|[\U000021a9-\U000021aa]"
    "|\U0000fe0f"  # variation selector-16
    "|\U0000200d"  # zero-width joiner
    "|\U000020e3"  # combining enclosing keycap
    ")+",
)

# ── Irodori-specific text rules (ported from podcast-note skill) ──

# Emojis Irodori-TTS v4.1-Small interprets as style annotations
# (irodori_tts/duration.py: ALLOWED_ANNOTATION_EMOJIS). Anything outside
# this list is not an annotation and only garbles the reading, so when
# keep_emoji=True we keep only these.
IRODORI_STYLE_EMOJI = (
    "⏩",
    "⏱️",
    "⏸️",
    "🌬️",
    "🍭",
    "🎛️",
    "🎭",
    "🎵",
    "🐢",
    "🐱",
    "👂",
    "👃",
    "👅",
    "👌",
    "👏",
    "💋",
    "💥",
    "💦",
    "💪",
    "📄",
    "📞",
    "📢",
    "📣",
    "📖",
    "😆",
    "😊",
    "😌",
    "😎",
    "😏",
    "😒",
    "😖",
    "😟",
    "😠",
    "😪",
    "😭",
    "😮",
    "😮‍💨",
    "😰",
    "😱",
    "😲",
    "😴",
    "🙄",
    "🙏",
    "🤐",
    "🤔",
    "🤢",
    "🤧",
    "🤭",
    "🥤",
    "🥱",
    "🥴",
    "🥵",
    "🥹",
    "🥺",
    "🫣",
    "🫶",
)
_RE_STYLE_EMOJI = re.compile("|".join(sorted((re.escape(e) for e in IRODORI_STYLE_EMOJI), key=len, reverse=True)))

_ONES = ["", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"]
_HUND = [
    "",
    "ひゃく",
    "にひゃく",
    "さんびゃく",
    "よんひゃく",
    "ごひゃく",
    "ろっぴゃく",
    "ななひゃく",
    "はっぴゃく",
    "きゅうひゃく",
]
_THOU = ["", "せん", "にせん", "さんぜん", "よんせん", "ごせん", "ろくせん", "ななせん", "はっせん", "きゅうせん"]
_KANSUJI = {c: i for i, c in enumerate("〇一二三四五六七八九")}


def _year_kana(n: int) -> str:
    """年号を読み仮名に。2003 → にせんさんねん（「二〇〇三年」は誤読するため）"""
    tens = n // 10 % 10
    return (
        _THOU[n // 1000]
        + _HUND[n // 100 % 10]
        + ("じゅう" if tens == 1 else _ONES[tens] + "じゅう" if tens else "")
        + _ONES[n % 10]
        + "ねん"
    )


def read_years(text: str) -> str:
    """4桁年号をかな化し、漢数字間の「・」を「てん」に変換する。"""

    def repl(m: re.Match) -> str:
        raw = m.group(1)
        digits = "".join(str(_KANSUJI[c]) if c in _KANSUJI else c for c in raw)
        return _year_kana(int(digits)) if digits.isdigit() else m.group(0)

    text = re.sub(r"([0-9〇一二三四五六七八九]{4})年", repl, text)
    return re.sub(r"(?<=[〇一二三四五六七八九十百千])・(?=[〇一二三四五六七八九])", "てん", text)


# Fleet-global pronunciation dictionary (TSV: 表記<TAB>読み), applied
# longest-first right before synthesis. Irodori has no furigana input, so
# this is the only lever against misread proper nouns.
_YOMI_FILENAME = "voice_yomi.tsv"
_yomi_cache: list[tuple[str, str]] | None = None
_yomi_mtime: float = 0.0


def load_yomi() -> list[tuple[str, str]]:
    """Load the yomi dictionary from the data dir, mtime-cached."""
    global _yomi_cache, _yomi_mtime

    from core.paths import get_data_dir

    path = get_data_dir() / _YOMI_FILENAME
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    if _yomi_cache is not None and mtime == _yomi_mtime:
        return _yomi_cache
    pairs: list[tuple[str, str]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or "\t" not in line:
                continue
            src, dst = line.split("\t", 1)
            if src.strip():
                pairs.append((src.strip(), dst.strip()))
    except OSError:
        return []
    _yomi_cache = sorted(pairs, key=lambda p: -len(p[0]))
    _yomi_mtime = mtime
    return _yomi_cache


def sanitize_for_tts(text: str, *, keep_emoji: bool = False) -> str:
    """Strip Markdown and HTML comments for TTS consumption.

    Emoji are stripped by default; pass ``keep_emoji=True`` for engines
    (Irodori) that read them as emotion cues and speak better with them
    (non-allowlist emoji are still removed).  Reading substitutions
    (yomi dict / year kana) are NOT applied here — the result is fit for
    subtitle display; run ``apply_reading_rules`` on the TTS-bound copy.
    """
    text = _RE_HTML_COMMENT.sub("", text)
    text = _RE_HTML_COMMENT_OPEN.sub("", text)
    text = _RE_MD_CODE_BLOCK.sub("", text)
    text = _RE_MD_HEADING.sub("", text)
    text = _RE_MD_BOLD.sub(r"\1", text)
    text = _RE_MD_ITALIC.sub(r"\1", text)
    text = _RE_MD_INLINE_CODE.sub(r"\1", text)
    text = _RE_MD_LINK.sub(r"\1", text)
    text = _RE_MD_LIST_BULLET.sub("", text)
    text = _RE_MD_LIST_NUMBERED.sub("", text)
    text = _RE_MD_TABLE_PIPE.sub("", text)
    text = _RE_MD_HR.sub("", text)
    if keep_emoji:
        # Keep only annotation emojis; anything else garbles the reading.
        text = _RE_EMOJI.sub(lambda m: "".join(_RE_STYLE_EMOJI.findall(m.group(0))), text)
    else:
        text = _RE_EMOJI.sub("", text)
    return text.strip()


def apply_reading_rules(text: str) -> str:
    """Apply pronunciation substitutions for Irodori synthesis input.

    Yomi-dict replacement and year kana conversion turn kanji into kana,
    which reads correctly but looks bad in subtitles — apply this only to
    the string sent to the TTS engine, never to the display copy.
    """
    for src, dst in load_yomi():
        text = text.replace(src, dst)
    return read_years(text)


def _wav_seconds(data: bytes) -> float | None:
    """Duration of a complete WAV blob, or None if *data* is not parseable WAV."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            rate = w.getframerate()
            return w.getnframes() / rate if rate else None
    except Exception:
        return None


def _normalized_rms_from_pcm16(audio_data: bytes) -> float:
    """Calculate normalized RMS from 16-bit mono PCM bytes."""
    if len(audio_data) < PCM16_BYTES_PER_SAMPLE:
        return 0.0
    sample_count = len(audio_data) // PCM16_BYTES_PER_SAMPLE
    if sample_count == 0:
        return 0.0
    samples = memoryview(audio_data).cast("h")
    # Downsample for large chunks to keep CPU usage low.
    step = 4 if sample_count > 64_000 else 1
    sum_sq = 0.0
    count = 0
    for i in range(0, sample_count, step):
        value = samples[i] / 32768.0
        sum_sq += value * value
        count += 1
    if count == 0:
        return 0.0
    return (sum_sq / count) ** 0.5


def _normalize_probe_text(text: str) -> str:
    """Normalize STT/TTS text to comparable letters and numbers."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in normalized if unicodedata.category(char)[0] in {"L", "M", "N"})


def _is_self_echo(text: str, recent: deque[str] | list[str] | tuple[str, ...]) -> bool:
    """Return whether *text* is substantially contained in recent TTS."""
    candidate = _normalize_probe_text(text)
    reference = _normalize_probe_text("".join(recent))
    if not candidate or not reference:
        return False
    if len(candidate) <= 2:
        candidate_units = set(candidate)
        reference_units = set(reference)
    else:
        candidate_units = {candidate[i : i + 2] for i in range(len(candidate) - 1)}
        reference_units = {reference[i : i + 2] for i in range(len(reference) - 1)}
    if not candidate_units:
        return False
    containment = len(candidate_units & reference_units) / len(candidate_units)
    return containment >= ECHO_SIMILARITY_THRESHOLD


# ── VoiceSession ────────────────────────────────────────────────


class VoiceSession:
    """Manages a single voice conversation session with an Anima."""

    def __init__(
        self,
        anima_name: str,
        ws: Any,
        stt: VoiceSTT,
        tts: BaseTTSProvider,
        tts_config: TTSConfig,
        supervisor: Any,
        voice_config: Any,
        front_model: str | None = None,
        front_api_base: str | None = None,
    ) -> None:
        """Initialize voice session.

        Args:
            anima_name: Target Anima name.
            ws: WebSocket (send_json, send_bytes).
            stt: STT engine.
            tts: TTS provider.
            tts_config: Per-session TTS config.
            supervisor: ProcessSupervisor for IPC.
            voice_config: Voice configuration (stt_refine_enabled, etc.).
            front_model: Optional voice front lane model name. Falls back to
                ``front_model`` on *voice_config* (None → legacy path).
            front_api_base: Optional OpenAI-compatible base URL for the front
                lane. Falls back to ``front_api_base`` on *voice_config*.
        """
        self._anima_name = anima_name
        self._ws = ws
        self._stt = stt
        self._tts = tts
        self._tts_config = tts_config
        self._supervisor = supervisor
        self._voice_config = voice_config
        self._audio_buffer: bytearray = bytearray()
        # Streaming STT: rolling re-decode with LocalAgreement-2. Decode is
        # synchronous; it is sheduled through run_in_executor so the event loop
        # is never blocked and only one decode is in flight at a time.
        self._streamer = StreamingTranscriber(
            lambda buf, initial_prompt: stt.transcribe_buffer(buf, initial_prompt=initial_prompt)
        )
        self._stream_task: asyncio.Task[None] | None = None
        self._streaming_busy = False
        self._finalizing = False
        self._tts_playing = False
        self._interrupted = False
        self._processing = False
        self._recent_tts_text: deque[str] = deque(maxlen=6)
        self._probe_active = False
        self._probe_started = 0.0
        self._probe_task: asyncio.Task[None] | None = None
        self._probe_followup_pending = False
        self._tts_available: bool | None = None
        self._splitter = StreamingSentenceSplitter()
        self._consecutive_tts_failures: int = 0
        self._tts_queue: asyncio.Queue[str] | None = None
        self._tts_worker: asyncio.Task[None] | None = None

        # ask_anima delegation state (PR-3). Queues/task are created lazily
        # on first use (inside the running event loop).
        self._delegation_jobs: dict[int, asyncio.Task] = {}
        self._delegation_job_counter: int = 0
        self._delegation_results: asyncio.Queue[str] | None = None
        self._delegation_done: asyncio.Queue[None] | None = None
        self._delegation_watcher: asyncio.Task | None = None

        # Proactive (silence-triggered) self-speech state. ``_last_activity``
        # tracks the newest user / session activity via ``time.monotonic()``;
        # when it ages past ``_proactive_delay`` the idle watcher may speak.
        self._last_activity: float = time.monotonic()
        self._proactive_count: int = 0
        # What the monologue already said (block-list for the next prompt).
        self._monologue_log: deque[str] = deque(maxlen=5)
        self._proactive_delay: float = float(getattr(voice_config, "proactive_initial_delay_sec", 10.0))
        self._proactive_lead_sec: float = float(getattr(voice_config, "proactive_lead_sec", 5.0))
        # Estimated monotonic time when the client finishes playing everything
        # sent so far. Synthesis finishing is *not* playback finishing.
        self._playback_end_at: float = 0.0
        self._idle_watcher: asyncio.Task | None = None
        self._idle_tick_sec: float = 5.0
        self._closed = False

        if front_model is None:
            front_model = getattr(voice_config, "front_model", None) or None
        if front_api_base is None:
            front_api_base = getattr(voice_config, "front_api_base", None) or None
        self._front_model = front_model or None
        self._front_api_base = front_api_base or None
        self._front_lane: Any | None = None

    async def handle_audio_chunk(self, data: bytes) -> None:
        """Receive audio chunk from browser, accumulate in buffer and feed the
        streaming transcriber (which may emit committed partials)."""
        # We are talking: whatever the mic hears is our own TTS leaking through
        # the speakers. The client suppresses it too, but its playback flag can
        # lag a frame or two — dropping here makes self-transcription impossible.
        if self._tts_playing and not self._probe_active:
            return
        if len(self._audio_buffer) + len(data) > MAX_AUDIO_BUFFER_BYTES:
            self._audio_buffer.clear()
            logger.warning("Audio buffer overflow (%s), cleared", self._anima_name)
        self._audio_buffer.extend(data)
        # Only *recognized speech* counts as activity. A hands-free (VAD) mic
        # streams silence continuously, so resetting on every raw chunk kept
        # the idle timer pinned at zero and made proactive speech unreachable.
        if self._streamer.committed:
            self._last_activity = time.monotonic()
        if self._streamer.feed(data):
            self._maybe_start_streaming_stt()

    def _maybe_start_streaming_stt(self) -> None:
        """Start the streaming decode loop if a decode is due and none is running
        already (prevents overlapping / double decodes)."""
        if (
            self._streaming_busy
            or self._stream_task is not None
            or (self._processing and not self._probe_active)
            or self._finalizing
        ):
            return
        if not self._streamer.ready():
            return
        self._streaming_busy = True
        self._stream_task = asyncio.create_task(self._stream_stt_loop(), name=f"stream-stt-{self._anima_name}")
        self._stream_task.add_done_callback(self._stream_stt_done)

    def _stream_stt_done(self, task: asyncio.Task) -> None:
        self._stream_task = None
        self._streaming_busy = False
        if not task.cancelled():
            try:
                task.result()
            except Exception:
                logger.debug("Streaming STT task error (%s)", self._anima_name, exc_info=True)
        # Catch up on audio that accumulated while we were busy.
        if not self._finalizing and (not self._processing or self._probe_active):
            self._maybe_start_streaming_stt()

    async def _stream_stt_loop(self) -> None:
        """Re-decode the rolling buffer off the event loop, emitting committed
        partials. Exits when caught up, finalizing, or processing a reply."""
        while True:
            if self._finalizing or (self._processing and not self._probe_active):
                break
            loop = asyncio.get_running_loop()
            committed = await loop.run_in_executor(None, self._streamer.run_decode)
            if committed:
                if self._probe_active:
                    await self._judge_probe(committed, final=False)
                else:
                    await self._ws.send_json({"type": "transcript_partial", "text": committed})
            if not self._streamer.ready():
                break

    async def handle_speech_end(self, from_person: str = "human") -> None:
        """Process accumulated audio: STT -> optional refine -> Chat -> TTS."""
        if self._probe_active:
            await self._finish_probe()
            return
        if self._probe_followup_pending:
            deadline = time.monotonic() + 2.0
            while self._processing and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            if self._processing:
                logger.warning("Probe follow-up still waiting for current turn (%s)", self._anima_name)
                return
            self._probe_followup_pending = False
            # The previous turn may have reset the rolling decoder after the
            # probe. Transcribe the preserved full buffer to avoid losing its
            # beginning.
            self._streamer.reset()
        if self._processing:
            logger.debug("speech_end ignored — already processing (%s)", self._anima_name)
            return
        # A real user turn resets the proactive state so the next silence
        # period begins with the conversational first prompt again.
        self._proactive_count = 0
        self._proactive_delay = float(getattr(self._voice_config, "proactive_initial_delay_sec", 10.0))
        self._processing = True
        try:
            await self._do_speech_end(from_person)
        finally:
            self._processing = False
            self._finalizing = False
            self._streamer.reset()
            self._last_activity = time.monotonic()

    async def _check_tts_health(self) -> bool:
        """Check TTS availability. Only caches positive results; retries on failure."""
        if self._tts_available:
            return True
        try:
            ok = await self._tts.health_check()
        except Exception:
            ok = False
        self._tts_available = ok
        if not ok:
            logger.warning(
                "TTS provider unavailable for %s (%s)",
                self._anima_name,
                self._tts_config.provider,
            )
            await self._ws.send_json({"type": "error", "message": "TTS unavailable"})
        return ok

    def invalidate_tts_health(self) -> None:
        """Reset cached TTS health so next speech_end rechecks."""
        self._tts_available = None

    async def _do_speech_end(self, from_person: str) -> None:
        """Inner speech_end logic, guarded by _processing flag."""
        self._recent_tts_text.clear()
        audio_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        if not audio_data:
            return
        if len(audio_data) < MIN_SPEECH_BYTES:
            logger.debug("Ignore short voice chunk: bytes=%s", len(audio_data))
            return
        rms = _normalized_rms_from_pcm16(audio_data)
        if rms < SILENCE_RMS_THRESHOLD:
            logger.debug("Ignore likely silence: rms=%.5f bytes=%s", rms, len(audio_data))
            return

        # Stop live streaming so finalize sees a stable buffer.
        self._finalizing = True
        if self._stream_task is not None:
            try:
                await self._stream_task
            except Exception:
                pass

        # 1. STT
        streaming_used = False
        try:
            if self._streamer.has_content():
                # Streaming path: finalize the rolling decode. The committed
                # prefix was shown live via transcript_partial; the remainder
                # is decoded here. Decode runs off the event loop.
                streaming_used = True
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(None, self._streamer.finalize)
                text = text.strip()
                language = self._streamer.last_language or "ja"
            else:
                # No streaming decode happened yet (very short input, or tests
                # pre-loading _audio_buffer directly). Keep the legacy full-
                # buffer transcription so the final transcript event stays
                # backward-compatible (existing tests pass unmodified).
                result = await self._stt.transcribe_buffer_async(audio_data)
                text = result.get("raw_text", "").strip()
                language = result.get("language", "ja") or "ja"
        except Exception as e:
            logger.exception("STT failed: %s", e)
            await self._send_error(t("voice.stt_failed"))
            return

        if not text:
            return

        # 2. Optional LLM refine (skipped on the streaming path so the
        # LocalAgreement-committed transcript is used as-is, per plan PR-1).
        if not streaming_used and getattr(self._voice_config, "stt_refine_enabled", False):
            try:
                from core.tools.transcribe import refine_with_llm

                loop = asyncio.get_running_loop()
                refined = await loop.run_in_executor(
                    None,
                    lambda: refine_with_llm(
                        text,
                        language=language,
                    ),
                )
                text = refined.get("refined_text", text)
            except Exception as e:
                logger.warning("STT refine failed, using raw: %s", e)

        # 3. Send transcript to client
        await self._ws.send_json({"type": "transcript", "text": text})

        # 4. Check TTS health before entering IPC loop
        tts_ok = await self._check_tts_health()

        # 5. Send to Anima via IPC (streaming)
        await self._ws.send_json({"type": "response_start"})
        self._tts_playing = True
        self._interrupted = False

        timeout = IPC_STREAM_TIMEOUT
        try:
            timeout_attr = getattr(
                getattr(self._voice_config, "_server_config", None),
                "ipc_stream_timeout",
                None,
            )
            if timeout_attr is not None:
                timeout = float(timeout_attr)
        except (TypeError, AttributeError):
            pass

        response_done_sent = False
        if tts_ok:
            await self._start_tts_worker()
        try:
            # Voice front lane: when configured and reachable, handle the turn
            # here and skip the full agent loop (fallback on health failure).
            if self._front_model:
                lane = self._get_or_create_front_lane()
                try:
                    front_ok = await lane.check_health()
                except Exception:
                    front_ok = False
                if front_ok:
                    response_done_sent = await self._run_front_turn(lane, text, from_person, tts_ok)
                    return
                logger.warning(
                    "voice front unavailable (%s) — falling back to process_message",
                    self._front_model,
                )

            async for ipc_response in self._supervisor.send_request_stream(
                anima_name=self._anima_name,
                method="process_message",
                params={
                    "message": text + VOICE_MODE_SUFFIX,
                    "from_person": from_person,
                    "intent": "",
                    "stream": True,
                    "voice_mode": True,
                    "images": [],
                    "attachment_paths": [],
                },
                timeout=timeout,
            ):
                if self._interrupted:
                    break

                if ipc_response.done:
                    result_data = ipc_response.result or {}
                    cycle_result = result_data.get("cycle_result", {})
                    emotion = cycle_result.get("emotion", "neutral")
                    remaining = self._splitter.flush()
                    if remaining and tts_ok:
                        await self._enqueue_tts(remaining)
                    await self._finish_tts_and_response_done(emotion)
                    response_done_sent = True
                    break

                if ipc_response.chunk:
                    try:
                        chunk_data = json.loads(ipc_response.chunk)
                    except json.JSONDecodeError:
                        chunk_data = {"type": "text_delta", "text": ipc_response.chunk}

                    if chunk_data.get("type") == "keepalive":
                        continue

                    if chunk_data.get("type") == "text_delta":
                        delta = chunk_data.get("text", "")
                        if delta:
                            await self._ws.send_json(
                                {
                                    "type": "response_text",
                                    "text": delta,
                                    "done": False,
                                }
                            )
                            if tts_ok:
                                sentences = self._splitter.feed(delta)
                                for sentence in sentences:
                                    if self._interrupted:
                                        break
                                    await self._enqueue_tts(sentence)

                    elif chunk_data.get("type") == "thinking_start":
                        await self._ws.send_json({"type": "thinking_status", "thinking": True})
                    elif chunk_data.get("type") == "thinking_end":
                        await self._ws.send_json({"type": "thinking_status", "thinking": False})
                    elif chunk_data.get("type") == "thinking_delta":
                        delta = chunk_data.get("text", "")
                        if delta:
                            await self._ws.send_json(
                                {
                                    "type": "thinking_delta",
                                    "text": delta,
                                }
                            )

                    elif chunk_data.get("type") == "cycle_done":
                        cycle_result = chunk_data.get("cycle_result", {})
                        emotion = cycle_result.get("emotion", "neutral")
                        remaining = self._splitter.flush()
                        if remaining and tts_ok:
                            await self._enqueue_tts(remaining)
                        await self._finish_tts_and_response_done(emotion)
                        response_done_sent = True
                        break

        except Exception as e:
            logger.exception("Voice session IPC error: %s", e)
            await self._send_error(str(e))
        finally:
            if not response_done_sent:
                try:
                    # Drain any enqueued audio before the fallback terminal frames
                    # unless barge-in already discarded the queue.
                    if tts_ok and not self._interrupted:
                        await self._drain_tts_queue()
                    await self._ws.send_json(
                        {
                            "type": "emotion",
                            "emotion": "neutral",
                        }
                    )
                    await self._ws.send_json(
                        {
                            "type": "response_done",
                            "emotion": "neutral",
                        }
                    )
                except Exception:
                    pass
            await self._stop_tts_worker()
            self._tts_playing = False
            self._interrupted = False
            self._splitter.flush()

    async def _start_tts_worker(self) -> None:
        """Start a single ordered TTS consumer for the current utterance."""
        await self._stop_tts_worker()
        self._tts_queue = asyncio.Queue(maxsize=TTS_QUEUE_MAXSIZE)
        self._tts_worker = asyncio.create_task(
            self._tts_consumer_loop(),
            name=f"tts-worker-{self._anima_name}",
        )

    async def _tts_consumer_loop(self) -> None:
        """Pull sentences in order and synthesize. One consumer preserves order."""
        queue = self._tts_queue
        if queue is None:
            return
        try:
            while True:
                sentence = await queue.get()
                try:
                    if not self._interrupted:
                        await self._synthesize_and_send(sentence)
                except Exception:
                    # Keep the session alive; synthesis errors are handled inside
                    # _synthesize_and_send, this is a last-resort guard.
                    logger.exception("TTS worker unexpected error (%s)", self._anima_name)
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            raise

    async def _enqueue_tts(self, sentence: str) -> None:
        """Producer side: enqueue a sentence without waiting for synthesis."""
        queue = self._tts_queue
        if not sentence or queue is None or self._interrupted:
            return
        await queue.put(sentence)

    async def _drain_tts_queue(self) -> None:
        """Wait until the consumer finishes every enqueued sentence."""
        queue = self._tts_queue
        if queue is None:
            return
        await queue.join()

    def _clear_tts_queue(self) -> None:
        """Discard pending sentences (barge-in). In-flight item is left to consumer."""
        queue = self._tts_queue
        if queue is None:
            return
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                queue.task_done()

    async def _stop_tts_worker(self) -> None:
        """Cancel consumer task and drop the queue (no leak on disconnect)."""
        worker = self._tts_worker
        queue = self._tts_queue
        self._tts_worker = None
        self._tts_queue = None
        if queue is not None:
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    queue.task_done()
        if worker is not None and not worker.done():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("TTS worker stop error", exc_info=True)

    async def _finish_tts_and_response_done(self, emotion: str) -> None:
        """Drain TTS then emit emotion + response_done in that order."""
        if not self._interrupted:
            await self._drain_tts_queue()
        await self._ws.send_json({"type": "emotion", "emotion": emotion})
        await self._ws.send_json({"type": "response_done", "emotion": emotion})

    # ── voice front lane ───────────────────────────────────────────

    def _get_or_create_front_lane(self) -> Any:
        """Lazily build and cache the single per-session voice front lane."""
        if self._front_lane is None:
            from core.paths import get_animas_dir
            from core.prompt.builder import build_voice_front_prompt
            from core.voice.front import VoiceFrontLane

            anima_dir = get_animas_dir() / self._anima_name
            system_prompt = build_voice_front_prompt(
                anima_dir,
                anima_name=self._anima_name,
            )
            self._front_lane = VoiceFrontLane(
                model=self._front_model,
                api_base=self._front_api_base or "",
                system_prompt=system_prompt,
            )
        return self._front_lane

    # ── proactive silence-triggered self-speech ───────────────────

    def _ensure_idle_watcher(self) -> None:
        """Start the idle watcher task once proactive speech is enabled.

        Requires ``proactive_enabled`` and a configured front lane; otherwise
        the task is never created.
        """
        if not getattr(self._voice_config, "proactive_enabled", False):
            return
        if not self._front_model:
            return
        if self._idle_watcher is None or self._idle_watcher.done():
            self._idle_watcher = asyncio.create_task(
                self._idle_watcher_loop(),
                name=f"idle-watcher-{self._anima_name}",
            )

    def _should_proactive(self, *, processing: bool | None = None) -> bool:
        """True when the idle watcher may run a proactive self-turn right now.

        Every fire-guard is re-checked so a self-turn never steps on an in-
        progress user turn, TTS playback, incoming speech, or delegation.
        ``processing`` lets the loop re-evaluate the other guards *after* it
        has already taken the processing lock itself (pass ``False`` then, so
        the lock it holds is not treated as a blocker).
        """
        if not getattr(self._voice_config, "proactive_enabled", False):
            return False
        if not self._front_model:
            return False
        if time.monotonic() - self._last_activity < self._proactive_delay:
            return False
        if self._tts_playing:
            return False
        # Wait for the client to nearly finish playing what it already has;
        # otherwise monologues pile up in its queue faster than it can speak.
        if time.monotonic() < self._playback_end_at - self._proactive_lead_sec:
            return False
        # Buffered raw audio is *not* a blocker: an open hands-free mic always
        # has some. Only a pending decode or already-recognized speech is.
        if self._streamer.ready() or self._streamer.committed or self._streaming_busy:
            return False
        if self._delegation_jobs:
            return False
        # Pending ask_anima results are surfaced by a dedicated delegation
        # self-turn / the next user turn, not by a silence turn (M2).
        if self._delegation_results is not None and not self._delegation_results.empty():
            return False
        busy = self._processing if processing is None else processing
        return not busy

    async def _idle_watcher_loop(self) -> None:
        """Poll for sustained silence and run a proactive self-turn when due.

        Successful self-turns repeat at a constant interval until the user
        responds, while the count selects progressively varied prompts.
        """
        from core.voice.front import READ_MEMORY_TOOL

        while not self._closed:
            await asyncio.sleep(self._idle_tick_sec)
            if not self._should_proactive():
                continue
            if self._closed:
                break
            # A proactive turn takes the processing lock so it can never share
            # the front lane / TTS worker with a greet or concurrent user
            # turn (C2), and release it via try/finally.
            self._processing = True
            try:
                lane = self._get_or_create_front_lane()
                try:
                    ok = await lane.check_health()
                except Exception:
                    ok = False
                if not ok:
                    # Back off a full delay window instead of poking a down
                    # lane on every tick (M3).
                    self._last_activity = time.monotonic()
                    continue
                # Re-check the remaining fire-guards after the (slow) health
                # probe; the processing lock is already held here.
                if not self._should_proactive(processing=False):
                    continue
                try:
                    turn_ok = await self._run_front_turn(
                        lane,
                        build_proactive_prompt(self._proactive_count, list(self._monologue_log)),
                        "human",
                        await self._check_tts_health(),
                        record_user=False,
                        record=False,
                        tools=[READ_MEMORY_TOOL],
                        tool_choice="required" if proactive_turn_uses_memory(self._proactive_count) else None,
                        keep_history=False,
                        max_tokens=MONOLOGUE_MAX_TOKENS,
                        temperature=MONOLOGUE_TEMPERATURE,
                        drain_results=False,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Proactive self-turn failed (%s)", self._anima_name)
                    turn_ok = False
                if turn_ok:
                    self._proactive_count += 1
                    spoken = lane.last_full_text if isinstance(lane.last_full_text, str) else ""
                    said = re.sub(r"<!--.*?-->", "", spoken, flags=re.DOTALL).strip()
                    if said:
                        self._monologue_log.append(said[:60])
                # Always rewind the idle timer — success spoke just now, and a
                # failure or down-lane must back off a full delay window.
                self._last_activity = time.monotonic()
            finally:
                self._processing = False

    async def _emit_text_delta(self, delta: str, tts_ok: bool) -> None:
        """Send a text delta to the client and feed the TTS sentence splitter."""
        await self._ws.send_json({"type": "response_text", "text": delta, "done": False})
        if tts_ok:
            sentences = self._splitter.feed(delta)
            for sentence in sentences:
                if self._interrupted:
                    break
                await self._enqueue_tts(sentence)

    def _record_front_conversation(
        self,
        user_text: str,
        response_text: str,
        from_person: str,
        *,
        record_user: bool = True,
    ) -> None:
        """Persist the front turn into the anima's default conversation.

        ``record_user=False`` (used by proactive self-turns) records only the
        assistant reply so the synthetic instruction never pollutes the history.

        Reuses the existing ``ConversationMemory`` record path so the turn is
        visible from the text chat; failures are non-fatal (front chat must
        stay available even if recording is unavailable).  The user turn uses
        ``from_person`` as the role, matching the existing chat path.

        Concurrency note: this runs in the *server* process and writes
        ``state/conversation.json`` via read-modify-write, so it can race
        with writes from the anima's own process (heartbeat etc.).  There is
        currently no supervisor IPC method to append a conversation turn from
        the server side, so the direct write is kept and the risk documented.
        """
        from core.memory.conversation import ConversationMemory

        try:
            from core.paths import get_animas_dir

            conversation = ConversationMemory(get_animas_dir() / self._anima_name, None)
            if record_user:
                conversation.append_turn(from_person or "human", user_text)
            conversation.append_turn("assistant", response_text)
            conversation.save()
        except Exception:
            logger.debug("Failed to persist front conversation (%s)", self._anima_name, exc_info=True)

    async def _run_front_turn(
        self,
        lane: Any,
        text: str,
        from_person: str,
        tts_ok: bool,
        *,
        record_user: bool = True,
        record: bool = True,
        tools: list | None = None,
        tool_choice: str | None = None,
        keep_history: bool = True,
        max_tokens: int | None = None,
        temperature: float | None = None,
        drain_results: bool = True,
    ) -> bool:
        """Stream one front-lane turn into TTS + WebSocket, then finish.

        ``record=False`` keeps proactive monologues out of conversation.json.
        ``record_user=False`` remains available to omit a synthetic user turn.
        ``drain_results=False`` keeps a silence turn from snatching pending
        delegation results (M2).

        Self-turns (proactive / delegation watcher) reach here without a live
        TTS worker; they own the full terminal-frame contract here (a leading
        ``response_start`` plus a guaranteed trailing ``response_done``), so the
        client never hangs and the subtitle starts a fresh bubble (H1/H2).
        User turns keep the worker and terminal frames that ``_do_speech_end``
        already owns.

        Returns ``True`` when the terminal response frames were emitted.
        """
        from core.voice.front import ASK_ANIMA_TOOL, READ_MEMORY_TOOL, extract_emotion

        if drain_results:
            results = self._drain_delegation_results()
            if results:
                text = f"{results}\n\n{text}"

        # Self-turns call this without a live TTS worker — without one
        # _enqueue_tts drops every sentence silently. Own the worker
        # lifecycle and the response_start/terminal frames here in that case;
        # user turns keep the worker (and frames) _do_speech_end started.
        owns_tts_worker = tts_ok and self._tts_queue is None
        if owns_tts_worker:
            await self._ws.send_json({"type": "response_start"})
            self._interrupted = False
            await self._start_tts_worker()
            self._tts_playing = True

        if tools is None:
            tools = [ASK_ANIMA_TOOL, READ_MEMORY_TOOL]

        lane.reset_turn()
        full: list[str] = []
        response_done_sent = False
        try:
            try:
                async for delta in lane.stream(
                    text,
                    tools=tools,
                    tool_executor=self._ask_anima,
                    tool_executors={
                        "ask_anima": lambda args: self._ask_anima(str(args.get("request", ""))),
                        "read_memory": self._read_memory,
                    },
                    tool_choice=tool_choice,
                    keep_history=keep_history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    if self._interrupted:
                        return False
                    await self._emit_text_delta(delta, tts_ok)
                    full.append(delta)
            except Exception as e:
                logger.exception("Voice front stream error: %s", e)
                await self._send_error(str(e))
                return False
            if self._interrupted:
                return False
            remaining = self._splitter.flush()
            if remaining and tts_ok:
                await self._enqueue_tts(remaining)
            full_text = "".join(full)
            if not full_text.strip():
                # Nothing was said (e.g. a thinking model spent the whole token
                # budget reasoning). Don't persist an empty turn or let a
                # proactive turn count it as a spoken one.
                logger.warning("Voice front turn produced no text (%s)", self._anima_name)
                return False
            if record:
                self._record_front_conversation(text, full_text, from_person, record_user=record_user)
            self._last_activity = time.monotonic()
            emotion = extract_emotion(full_text)
            await self._finish_tts_and_response_done(emotion)
            response_done_sent = True
            return True
        finally:
            if owns_tts_worker:
                if not response_done_sent:
                    # A self-turn must always terminate cleanly (barge-in / a
                    # stream failure): drop any partial splitter content and
                    # emit neutral terminal frames so the client state never
                    # hangs on a half-spoken turn.
                    self._splitter.flush()
                    try:
                        await self._ws.send_json({"type": "emotion", "emotion": "neutral"})
                        await self._ws.send_json({"type": "response_done", "emotion": "neutral"})
                    except Exception:
                        pass
                await self._stop_tts_worker()
                self._tts_playing = False

    # ── ask_anima async delegation (PR-3) ─────────────────────────

    def _read_memory(self, args: dict) -> str:
        """Read this Anima's file-backed memory for the front lane."""
        from core.paths import get_animas_dir

        return read_memory_snippets(
            get_animas_dir() / self._anima_name,
            str(args.get("query", "")),
            page=self._proactive_count // len(_MONOLOGUE_CORNERS_JA),
        )

    def _ensure_delegation_state(self) -> None:
        """Create delegation queues and start the result watcher once."""
        if self._delegation_results is None:
            self._delegation_results = asyncio.Queue()
            self._delegation_done = asyncio.Queue()
        if self._delegation_watcher is None or self._delegation_watcher.done():
            self._delegation_watcher = asyncio.create_task(
                self._delegation_watcher_loop(),
                name=f"ask-anima-watcher-{self._anima_name}",
            )

    def _ask_anima(self, request: str) -> str:
        """Handle ``ask_anima`` from the front lane (synchronous tool).

        Matches the ``tool_executor`` interface ``(request_text) -> str``.
        Fires a fire-and-forget ``asyncio.Task`` that runs the request through
        the full agent loop (``process_message``) and returns an ACK string
        immediately so the front conversation is not blocked. At most
        ``MAX_ASK_ANIMA_CONCURRENT`` jobs run at once.
        """
        request = (request or "").strip()
        if not request:
            request = "（依頼内容が指定されていません）"
        if len(self._delegation_jobs) >= MAX_ASK_ANIMA_CONCURRENT:
            return f"実行中の依頼が{MAX_ASK_ANIMA_CONCURRENT}件ある。完了を待ってほしい"
        self._ensure_delegation_state()
        self._delegation_job_counter += 1
        job = self._delegation_job_counter
        task = asyncio.create_task(
            self._run_ask_anima_job(job, request),
            name=f"ask-anima-{self._anima_name}-{job}",
        )
        self._delegation_jobs[job] = task
        return f"受理しました (job {job})。完了したら知らせます"

    async def _run_ask_anima_job(self, job: int, request: str) -> None:
        """Consume the full-agent stream for one delegated request and surface
        the result back to the front lane's reflow queue."""
        result_text = ""
        try:
            async for resp in self._supervisor.send_request_stream(
                anima_name=self._anima_name,
                method="process_message",
                params={
                    "message": request + ASK_ANIMA_DELEGATION_NOTE,
                    "from_person": "human",
                    "intent": "",
                    "stream": True,
                    # Full-capability lane: the delegated job must NOT be
                    # downgraded by voice_mode (voice_thinking_effort etc.).
                    "voice_mode": False,
                    "images": [],
                    "attachment_paths": [],
                },
                timeout=IPC_STREAM_TIMEOUT,
            ):
                if getattr(resp, "done", False):
                    result_data = getattr(resp, "result", None) or {}
                    cycle_result = result_data.get("cycle_result", {}) or {}
                    summary = str(cycle_result.get("summary", "") or "")
                    if summary:
                        result_text = summary
                elif getattr(resp, "chunk", None):
                    try:
                        cd = json.loads(resp.chunk)
                    except (json.JSONDecodeError, TypeError):
                        cd = {}
                    if cd.get("type") == "cycle_done":
                        cycle_result = cd.get("cycle_result", {}) or {}
                        summary = str(cycle_result.get("summary", "") or "")
                        if summary:
                            result_text = summary
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # surface the failure so front can report it
            logger.exception("ask_anima job %s failed (%s): %s", job, self._anima_name, exc)
            result_text = "処理に失敗しました。詳細はログを確認してほしい"
        finally:
            self._delegation_jobs.pop(job, None)
            message = f"[ask_anima完了 job {job}: {result_text[:ASK_ANIMA_MAX_RESULT_CHARS]}]"
            if self._delegation_results is not None:
                await self._delegation_results.put(message)
            if self._delegation_done is not None:
                await self._delegation_done.put(None)

    def _drain_delegation_results(self) -> str:
        """Collect all completed ask_anima results for injection into the next
        user turn. Returns an empty string when nothing is pending."""
        if self._delegation_results is None:
            return ""
        parts: list[str] = []
        while True:
            try:
                parts.append(self._delegation_results.get_nowait())
            except asyncio.QueueEmpty:
                break
        return "\n".join(parts)

    async def _delegation_watcher_loop(self) -> None:
        """Proactively run a self-turn when an ask_anima result completes while
        no user turn / TTS playback is in progress, so the front reports the
        outcome in its own words. If a user turn is active, results stay in
        the queue for the next user turn instead (no double-reporting)."""
        while not self._closed:
            try:
                await self._delegation_done.get()
            except asyncio.CancelledError:
                return
            # Let a queued user turn (which also drains results) win if it is
            # about to start, avoiding a self-turn in the same breath.
            await asyncio.sleep(0.05)
            if self._processing or self._tts_playing:
                continue
            if self._closed or self._front_lane is None or not self._front_model:
                continue
            pref = self._drain_delegation_results()
            if not pref:
                continue
            lane = self._front_lane
            try:
                ok = await lane.check_health()
            except Exception:
                ok = False
            if not ok:
                continue
            synthetic = f"{pref} この結果を自分の言葉で短く報告して"
            try:
                await self._run_front_turn(lane, synthetic, "human", await self._check_tts_health())
            except Exception:
                logger.exception("ask_anima self-turn failed (%s)", self._anima_name)

    async def close(self) -> None:
        """Cancel TTS worker on session teardown (WS disconnect).

        Running ask_anima delegation tasks are deliberately NOT cancelled —
        they keep the full-agent loop going and the result is persisted in the
        conversation by ``process_message`` itself. The watcher (self-turn)
        is stopped because the WS is gone.
        """
        self._closed = True
        self._cancel_probe_timeout()
        watcher = self._delegation_watcher
        self._delegation_watcher = None
        if watcher is not None and not watcher.done():
            watcher.cancel()
            try:
                await watcher
            except (asyncio.CancelledError, Exception):
                pass
        idle = self._idle_watcher
        self._idle_watcher = None
        if idle is not None and not idle.done():
            idle.cancel()
            try:
                await idle
            except (asyncio.CancelledError, Exception):
                pass
        self._interrupted = True
        self._clear_tts_queue()
        await self._stop_tts_worker()
        task = self._stream_task
        self._stream_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _note_playback(self, seconds: float) -> None:
        """Extend the estimated client playback end by *seconds* of audio just sent."""
        now = time.monotonic()
        self._playback_end_at = max(self._playback_end_at, now) + max(seconds, 0.0)

    async def _synthesize_and_send(self, text: str) -> None:
        """TTS synthesize a sentence and send audio to client."""
        keep_emoji = getattr(self._tts_config, "provider", "") == "irodori"
        text = sanitize_for_tts(text, keep_emoji=keep_emoji)
        if not text:
            return
        # Subtitle keeps the original kanji; only the TTS input gets
        # yomi/kana substitutions (kana-heavy text is hard to read).
        spoken = apply_reading_rules(text) if keep_emoji else text
        try:
            # text rides along so the client can show a playback-synced subtitle
            self._recent_tts_text.append(text)
            await self._ws.send_json({"type": "tts_start", "text": text})
            secs = 0.0
            async for audio_chunk in self._tts.synthesize(spoken, self._tts_config):
                if self._interrupted:
                    break
                await self._ws.send_bytes(audio_chunk)
                secs += _wav_seconds(audio_chunk) or 0.0
            # ponytail: non-WAV (mp3 stream) falls back to ~6 chars/sec
            self._note_playback(secs or len(text) / 6.0)
            await self._ws.send_json({"type": "tts_done"})
            self._consecutive_tts_failures = 0
        except TTSSynthesisError as e:
            self._consecutive_tts_failures += 1
            logger.warning("TTS synthesis failed (%d consecutive): %s", self._consecutive_tts_failures, e)
            if self._consecutive_tts_failures >= 3:
                self.invalidate_tts_health()
            try:
                await self._ws.send_json({"type": "tts_error", "message": "TTS synthesis failed"})
                await self._ws.send_json({"type": "tts_done"})
            except Exception:
                pass
        except Exception as e:
            logger.warning("TTS send error: %s", e)
            try:
                await self._ws.send_json({"type": "tts_done"})
            except Exception:
                pass

    async def greet_and_speak(self) -> None:
        """Greet on connect — instant audio feedback that also warms the
        anima's chat runner / priming caches before the first utterance.
        Uses the cached greet path (1h cooldown), so repeated popups are cheap."""
        try:
            # 90s: cold greet = runner spawn + generate (~25s) + slow-exit
            # kill-wait (30s) can exceed 60s. Cached greets return instantly.
            result = await self._supervisor.send_request(
                anima_name=self._anima_name,
                method="greet",
                params={},
                timeout=90.0,
            )
        except Exception as e:
            logger.info("Voice greet skipped (%s): %s", self._anima_name, e)
            return
        text = str((result or {}).get("response", "")).strip()
        if not text or self._processing:
            return
        emotion = (result or {}).get("emotion", "neutral")
        tts_ok = await self._check_tts_health()
        self._interrupted = False
        try:
            await self._ws.send_json({"type": "response_start"})
            await self._ws.send_json({"type": "response_text", "text": text})
            await self._ws.send_json({"type": "emotion", "emotion": emotion})
            if tts_ok:
                self._tts_playing = True
                # Same prefetch worker as speech replies — first audio still
                # arrives after the first sentence synthesizes, later ones pipeline.
                from core.voice.sentence_splitter import split_sentences

                await self._start_tts_worker()
                for sentence in split_sentences(text):
                    if self._interrupted or self._processing:
                        break
                    await self._enqueue_tts(sentence)
                if not self._interrupted and not self._processing:
                    await self._drain_tts_queue()
            await self._ws.send_json({"type": "response_done", "emotion": emotion})
        except Exception as e:
            logger.debug("Voice greet delivery failed (%s): %s", self._anima_name, e)
        finally:
            await self._stop_tts_worker()
            self._tts_playing = False
            self._last_activity = time.monotonic()
            # Start the idle watcher only after the greet is finished, so a
            # cold greet (potentially ~90s) can never race a proactive turn
            # for the front lane / TTS worker (C2).
            self._ensure_idle_watcher()

    async def handle_interrupt(self) -> None:
        """Handle barge-in: stop TTS, drop queued sentences, prepare for new STT."""
        self._probe_active = False
        self._cancel_probe_timeout()
        self._interrupted = True
        # Reopen the mic immediately — handle_audio_chunk drops input while we
        # are talking, and the turn's own finally may be a beat behind.
        self._tts_playing = False
        self._playback_end_at = 0.0
        self._audio_buffer.clear()
        self._streamer.reset()
        self._clear_tts_queue()

    async def handle_barge_probe(self) -> None:
        """Accept mic audio while TTS plays and request an STT verdict."""
        self._cancel_probe_timeout()
        self._probe_active = True
        self._probe_started = time.monotonic()
        self._probe_followup_pending = False
        self._audio_buffer.clear()
        self._streamer.reset()
        self._probe_task = asyncio.create_task(
            self._probe_timeout(),
            name=f"barge-probe-timeout-{self._anima_name}",
        )

    async def _probe_timeout(self) -> None:
        try:
            await asyncio.sleep(PROBE_TIMEOUT_SEC)
            if self._probe_active:
                await self._judge_probe("")
        except asyncio.CancelledError:
            pass

    def _cancel_probe_timeout(self) -> None:
        task = self._probe_task
        self._probe_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def _finish_probe(self) -> None:
        """Finalize probe audio at speech_end and process a confirmed turn."""
        audio_data = bytes(self._audio_buffer)
        self._finalizing = True
        task = self._stream_task
        if task is not None and task is not asyncio.current_task():
            try:
                await task
            except Exception:
                pass
        if not self._probe_active:
            self._finalizing = False
            return
        try:
            if self._streamer.has_content():
                loop = asyncio.get_running_loop()
                text = (await loop.run_in_executor(None, self._streamer.finalize)).strip()
            elif audio_data:
                result = await self._stt.transcribe_buffer_async(audio_data)
                text = result.get("raw_text", "").strip()
            else:
                text = ""
        except Exception:
            logger.exception("Probe STT failed (%s)", self._anima_name)
            text = ""
        finally:
            self._finalizing = False
        interrupted = await self._judge_probe(text)
        if interrupted:
            await self.handle_speech_end()

    async def _judge_probe(self, text: str, *, final: bool = True) -> bool:
        """Resolve an active probe exactly once and notify the browser.

        A streaming partial (``final=False``) that is still too short to judge
        leaves the probe open: "うん" may be the start of "うん、ちょっと待って".
        """
        if not self._probe_active:
            return False
        normalized = _normalize_probe_text(text)
        if not final and len(normalized) < PROBE_MIN_CHARS:
            return False
        self._probe_active = False
        self._cancel_probe_timeout()
        if len(normalized) < PROBE_MIN_CHARS or _is_self_echo(text, self._recent_tts_text):
            self._audio_buffer.clear()
            self._streamer.reset()
            await self._ws.send_json({"type": "barge_verdict", "interrupt": False})
            return False

        preserved_audio = bytes(self._audio_buffer)
        await self.handle_interrupt()
        self._audio_buffer.extend(preserved_audio)
        self._probe_followup_pending = True
        await self._ws.send_json({"type": "barge_verdict", "interrupt": True})
        return True

    async def handle_discard_audio(self) -> None:
        """Drop buffered mic input without touching the current turn.

        Sent when a VAD misfire aborted a recording: the noise must not leak
        into the next utterance, but an in-flight reply must keep streaming
        (that is what ``handle_interrupt`` is for).
        """
        self._audio_buffer.clear()
        self._streamer.reset()

    async def _send_error(self, message: str) -> None:
        """Send error message to client."""
        try:
            await self._ws.send_json({"type": "error", "message": message})
        except Exception:
            logger.debug("Failed to send error to client", exc_info=True)
