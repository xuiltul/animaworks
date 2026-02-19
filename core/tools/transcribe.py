# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""AnimaWorks transcribe tool -- Whisper speech-to-text with LLM refinement.

Uses faster-whisper for transcription and OllamaClient for text refinement.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "transcribe": {"expected_seconds": 120, "background_eligible": True},
}

# Configuration from environment
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE_TYPE", "default")
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "ja")
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:20b")

# Prompts directory (shipped alongside this module or from source project)
_PROMPTS_DIR = Path(__file__).parent.parent.parent / "templates" / "prompts" / "transcribe"

# Fallback: check TRANSCRIBE_PROMPTS_DIR environment variable
if not _PROMPTS_DIR.exists():
    _env_dir = os.environ.get("TRANSCRIBE_PROMPTS_DIR")
    if _env_dir and Path(_env_dir).exists():
        _PROMPTS_DIR = Path(_env_dir)

# ---------------------------------------------------------------------------
# Language-specific prompt loading
# ---------------------------------------------------------------------------

_prompt_cache: dict[str, dict] = {}


def _load_prompt(lang: str) -> dict:
    """Load language-specific prompt (with cache)."""
    if lang in _prompt_cache:
        return _prompt_cache[lang]

    lang_base = lang.split("-")[0].lower() if lang else DEFAULT_LANGUAGE

    prompt_file = _PROMPTS_DIR / f"{lang_base}.json"
    if not prompt_file.exists():
        for fallback in ("en", DEFAULT_LANGUAGE):
            prompt_file = _PROMPTS_DIR / f"{fallback}.json"
            if prompt_file.exists():
                break

    if not prompt_file.exists():
        # Minimal fallback prompt if no file available
        data = {
            "system_prompt": "You are a text refinement assistant. Clean up the following speech transcription, fixing punctuation and formatting while preserving the original meaning.",
            "user_template": "```\n{text}\n```",
        }
        _prompt_cache[lang] = data
        _prompt_cache[lang_base] = data
        return data

    with open(prompt_file, encoding="utf-8") as f:
        data = json.load(f)

    _prompt_cache[lang] = data
    _prompt_cache[lang_base] = data
    return data


# ---------------------------------------------------------------------------
# Whisper model singleton
# ---------------------------------------------------------------------------

_whisper_model = None


def _get_whisper_model():
    """Get Whisper model singleton (loaded on first use)."""
    global _whisper_model
    if _whisper_model is None:
        if WhisperModel is None:
            raise ImportError(
                "transcribe tool requires 'faster-whisper'. "
                "Install with: pip install animaworks[transcribe]"
            )
        device = WHISPER_DEVICE
        if device == "auto":
            device = "cuda" if shutil.which("nvidia-smi") else "cpu"
        compute = WHISPER_COMPUTE
        if compute == "default":
            compute = "float16" if device == "cuda" else "int8"
        _whisper_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
    return _whisper_model


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def transcribe(
    audio_path: str,
    language: str | None = None,
    vad_filter: bool = True,
) -> dict:
    """Transcribe audio using faster-whisper.

    Args:
        audio_path: Path to audio file.
        language: Language code (e.g. 'ja', 'en'). Auto-detected if None.
        vad_filter: Whether to apply Voice Activity Detection filtering.

    Returns:
        Dict with raw_text, language, duration, timing info, and segments.
    """
    if WhisperModel is None:
        raise ImportError(
            "transcribe tool requires 'faster-whisper'. "
            "Install with: pip install animaworks[transcribe]"
        )

    t0 = time.time()
    model = _get_whisper_model()
    load_time = time.time() - t0

    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=language,
        vad_filter=vad_filter,
    )
    segments_list = list(segments)
    transcribe_time = time.time() - t0

    raw_text = " ".join(seg.text.strip() for seg in segments_list)

    return {
        "raw_text": raw_text,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "load_time": load_time,
        "transcribe_time": transcribe_time,
        "speed": info.duration / transcribe_time if transcribe_time > 0 else 0,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments_list
        ],
    }


def refine_with_llm(
    raw_text: str,
    model: str | None = None,
    language: str = DEFAULT_LANGUAGE,
    custom_prompt: str | None = None,
    context_hint: str | None = None,
) -> dict:
    """Refine transcribed text using a local Ollama LLM via OllamaClient.

    Args:
        raw_text: Raw transcription text to refine.
        model: Ollama model name. Defaults to LLM_MODEL env var.
        language: Language code for prompt selection.
        custom_prompt: Additional user instruction appended to system prompt.
        context_hint: Context information appended to system prompt.

    Returns:
        Dict with refined_text, model name, and refine_time.
    """
    from core.tools.local_llm import OllamaClient

    model = model or DEFAULT_LLM_MODEL

    prompt_data = _load_prompt(language)
    system_prompt = prompt_data["system_prompt"]

    if context_hint:
        prefix = prompt_data.get("context_prefix", "Context:")
        system_prompt = f"{system_prompt}\n\n{prefix}\n{context_hint}"
    if custom_prompt:
        prefix = prompt_data.get("custom_prompt_prefix", "Additional instructions:")
        system_prompt = f"{system_prompt}\n\n{prefix} {custom_prompt}"

    # Build messages with few-shot examples
    messages: list[dict[str, str]] = []
    for shot in prompt_data.get("few_shot", []):
        messages.append({"role": "user", "content": shot["user"]})
        messages.append({"role": "assistant", "content": shot["assistant"]})

    user_template = prompt_data.get("user_template", "```\n{text}\n```")
    messages.append({"role": "user", "content": user_template.format(text=raw_text)})

    t0 = time.time()
    client = OllamaClient(model=model)
    refined = client.chat(
        messages=messages,
        system=system_prompt,
        temperature=0.1,
        max_tokens=8192,
        think="off",
    )
    refine_time = time.time() - t0

    # Guard: if refined text is drastically shorter, the LLM likely hallucinated
    raw_len = len(raw_text)
    refined_len = len(refined)
    if raw_len > 0 and refined_len < raw_len * 0.4:
        logger.warning(
            "Refinement too short (%d -> %d chars), falling back to raw text",
            raw_len,
            refined_len,
        )
        refined = raw_text

    return {
        "refined_text": refined,
        "model": model,
        "refine_time": refine_time,
    }


def process_audio(
    audio_path: str,
    language: str | None = None,
    model: str | None = None,
    raw_only: bool = False,
    custom_prompt: str | None = None,
    quiet: bool = False,
) -> dict:
    """Full pipeline: transcribe audio then optionally refine with LLM.

    Args:
        audio_path: Path to audio file.
        language: Language code (auto-detected if None).
        model: Ollama model for refinement.
        raw_only: If True, skip LLM refinement.
        custom_prompt: Additional refinement instructions.
        quiet: Suppress progress output.

    Returns:
        Dict with transcription and refinement results.
    """
    if not quiet:
        print(f"[1/2] Transcribing: {audio_path}", file=sys.stderr)

    whisper_result = transcribe(audio_path, language=language)

    if not quiet:
        print(
            f"  -> {whisper_result['duration']:.1f}s audio, "
            f"{whisper_result['speed']:.1f}x realtime, "
            f"lang={whisper_result['language']}",
            file=sys.stderr,
        )

    result: dict[str, Any] = {**whisper_result, "refined_text": None, "refine_time": 0}

    if not raw_only and whisper_result["raw_text"].strip():
        effective_model = model or DEFAULT_LLM_MODEL
        if not quiet:
            print(f"[2/2] Refining with {effective_model}...", file=sys.stderr)

        llm_result = refine_with_llm(
            whisper_result["raw_text"],
            model=model,
            language=whisper_result.get("language", language or DEFAULT_LANGUAGE),
            custom_prompt=custom_prompt,
        )
        result["refined_text"] = llm_result["refined_text"]
        result["refine_time"] = llm_result["refine_time"]
        result["refine_model"] = llm_result["model"]

        if not quiet:
            total = (
                whisper_result["load_time"]
                + whisper_result["transcribe_time"]
                + llm_result["refine_time"]
            )
            print(
                f"  -> Refine: {llm_result['refine_time']:.1f}s, "
                f"Total: {total:.1f}s",
                file=sys.stderr,
            )

    return result


# ---------------------------------------------------------------------------
# Tool schemas for agent integration
# ---------------------------------------------------------------------------


def get_tool_schemas() -> list[dict]:
    """Return JSON schemas for transcription tools."""
    return [
        {
            "name": "transcribe_audio",
            "description": (
                "Transcribe an audio file to text using faster-whisper, "
                "optionally refining the output with a local LLM."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Path to the audio file.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (e.g. 'ja', 'en'). Auto-detected if omitted.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Ollama model for LLM refinement (optional).",
                    },
                    "raw_only": {
                        "type": "boolean",
                        "description": "If true, return raw transcription without LLM refinement.",
                        "default": False,
                    },
                    "custom_prompt": {
                        "type": "string",
                        "description": "Additional instruction for LLM refinement (optional).",
                    },
                },
                "required": ["audio_path"],
            },
        },
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cli_main(argv: list[str] | None = None) -> None:
    """Standalone CLI for audio transcription."""
    parser = argparse.ArgumentParser(
        prog="animaworks-transcribe",
        description="AnimaWorks transcribe tool -- audio to text via faster-whisper + LLM.",
    )
    sub = parser.add_subparsers(dest="command")

    p_trans = sub.add_parser("transcribe", help="Transcribe an audio file")
    p_trans.add_argument("audio_path", help="Path to audio file")
    p_trans.add_argument(
        "-l", "--language", default=None,
        help="Language code (e.g. ja, en). Auto-detected if omitted.",
    )
    p_trans.add_argument(
        "-m", "--model", default=None,
        help=f"Ollama model for refinement (default: {DEFAULT_LLM_MODEL})",
    )
    p_trans.add_argument(
        "--raw", action="store_true",
        help="Skip LLM refinement, output raw transcription only.",
    )
    p_trans.add_argument(
        "-p", "--prompt", default=None,
        help="Custom refinement instruction.",
    )
    p_trans.add_argument(
        "-o", "--output", default="text", choices=["text", "json"],
        help="Output format (default: text).",
    )
    p_trans.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress progress messages.",
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "transcribe":
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            print(f"Error: File not found: {audio_path}", file=sys.stderr)
            sys.exit(1)

        result = process_audio(
            str(audio_path),
            language=args.language,
            model=args.model,
            raw_only=args.raw,
            custom_prompt=args.prompt,
            quiet=args.quiet,
        )

        if args.output == "json":
            output = {
                "text": result["refined_text"] or result["raw_text"],
                "raw_text": result["raw_text"],
                "language": result["language"],
                "duration": result["duration"],
                "transcribe_time": round(result["transcribe_time"], 2),
                "refine_time": round(result["refine_time"], 2),
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            final = result["refined_text"] or result["raw_text"]
            print(final)


# ── Dispatch ──────────────────────────────────────────


def dispatch(tool_name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call to the appropriate handler."""
    if tool_name == "transcribe_audio":
        return process_audio(
            audio_path=args["audio_path"],
            language=args.get("language"),
            model=args.get("model"),
            raw_only=args.get("raw_only", False),
            custom_prompt=args.get("custom_prompt"),
        )
    raise ValueError(f"Unknown tool: {tool_name}")


if __name__ == "__main__":
    cli_main()