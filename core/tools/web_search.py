# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Brave Web Search tool for AnimaWorks.

Migrated from ~/.local/bin/websearch.
Uses httpx instead of urllib.request.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from typing import Any

import httpx

from core.tools._base import ToolConfigError, get_credential, logger

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "search": {"expected_seconds": 10, "background_eligible": False},
}

# ---------------------------------------------------------------------------
# Brave Search API endpoint
# ---------------------------------------------------------------------------
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Language code mapping (Brave uses "jp" instead of "ja")
_LANG_MAP = {"ja": "jp"}


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------

def search(
    query: str,
    count: int = 10,
    lang: str = "ja",
    country: str = "US",
    freshness: str | None = None,
) -> list[dict[str, str]]:
    """Call the Brave Search API and return a list of result dicts.

    Each dict contains ``title``, ``url``, and ``description`` keys.

    Args:
        query: The search query string.
        count: Number of results to return (1-20).
        lang: Search language code (e.g. "ja", "en").
        country: Country code (e.g. "US", "JP").
        freshness: Optional freshness filter ("pd", "pw", "pm", "py").

    Returns:
        A list of dicts with title/url/description fields.

    Raises:
        ToolConfigError: If BRAVE_API_KEY is not set.
        httpx.HTTPStatusError: On non-2xx API responses.
    """
    api_key = get_credential("brave", "web_search", env_var="BRAVE_API_KEY")

    # Normalize language code for Brave API
    search_lang = _LANG_MAP.get(lang, lang)

    params: dict[str, Any] = {
        "q": query,
        "count": min(max(count, 1), 20),
        "search_lang": search_lang,
        "country": country,
    }
    if freshness:
        params["freshness"] = freshness

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    response = httpx.get(
        BRAVE_SEARCH_URL,
        params=params,
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()

    data = response.json()

    # Extract and normalize web results
    results: list[dict[str, str]] = []
    for item in data.get("web", {}).get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("description", ""),
        })

    return results


# ---------------------------------------------------------------------------
# Text formatting helpers
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities."""
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def format_results(results: list[dict[str, str]]) -> str:
    """Format search results as human-readable text."""
    if not results:
        return "No results found."

    lines: list[str] = []
    for i, item in enumerate(results, 1):
        title = item.get("title", "No title")
        url = item.get("url", "")
        desc = _strip_html(item.get("description", "No description"))
        lines.append(f"{i}. {title}")
        lines.append(f"   {url}")
        lines.append(f"   {desc}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anthropic tool_use schema
# ---------------------------------------------------------------------------

def get_tool_schemas() -> list[dict]:
    """Return Anthropic tool_use schemas for the web_search tool."""
    return [
        {
            "name": "web_search",
            "description": (
                "Search the web using the Brave Search API. "
                "Returns a list of results with title, URL, and description."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results (1-20, default 10).",
                        "default": 10,
                    },
                    "lang": {
                        "type": "string",
                        "description": "Search language code (e.g. 'ja', 'en').",
                        "default": "ja",
                    },
                    "freshness": {
                        "type": "string",
                        "description": (
                            "Freshness filter: 'pd' (past day), 'pw' (past week), "
                            "'pm' (past month), 'py' (past year)."
                        ),
                        "enum": ["pd", "pw", "pm", "py"],
                    },
                },
                "required": ["query"],
            },
        }
    ]


# ── Dispatch ──────────────────────────────────────────

def dispatch(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call by schema name."""
    if name == "web_search":
        return search(**args)
    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli_main(argv: list[str] | None = None) -> None:
    """Thin CLI entry point for web_search.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        description="Search the web using Brave Search API",
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=10,
        help="Number of results (1-20, default: 10)",
    )
    parser.add_argument(
        "-l", "--lang",
        default="ja",
        help="Search language (e.g. ja, en)",
    )
    parser.add_argument(
        "-f", "--freshness",
        choices=["pd", "pw", "pm", "py"],
        help="Freshness filter: pd=24h, pw=1week, pm=1month, py=1year",
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args(argv)

    results = search(
        query=args.query,
        count=args.count,
        lang=args.lang,
        freshness=args.freshness,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(format_results(results))