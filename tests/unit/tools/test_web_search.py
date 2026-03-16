"""Tests for core/tools/web_search.py — Brave Web Search tool."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from core.tools._base import ToolConfigError
from core.tools.web_search import (
    BRAVE_SEARCH_URL,
    _strip_html,
    dispatch,
    format_results,
    get_tool_schemas,
    search,
)


# ── Helper fixtures ───────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _set_brave_api_key():
    with patch("core.tools.web_search.get_credential", return_value="test-brave-key-123"):
        yield


def _make_brave_response(results: list[dict] | None = None) -> httpx.Response:
    """Build a mock httpx.Response with Brave search results."""
    if results is None:
        results = [
            {"title": "Result 1", "url": "https://example.com/1", "description": "Desc 1"},
            {"title": "Result 2", "url": "https://example.com/2", "description": "Desc <b>2</b>"},
        ]
    data = {"web": {"results": results}}
    return httpx.Response(200, json=data, request=httpx.Request("GET", BRAVE_SEARCH_URL))


# ── search() ──────────────────────────────────────────────────────


class TestSearch:
    def test_successful_search(self):
        mock_resp = _make_brave_response()
        with patch("core.tools.web_search.httpx.get", return_value=mock_resp):
            results = search("python programming")
        assert len(results) == 2
        assert results[0]["title"] == "Result 1"
        assert results[0]["url"] == "https://example.com/1"

    def test_search_params_construction(self):
        mock_resp = _make_brave_response([])
        with patch("core.tools.web_search.httpx.get", return_value=mock_resp) as mock_get:
            search("test query", count=5, lang="en", country="JP", freshness="pw")
        call_kwargs = mock_get.call_args
        params = call_kwargs.kwargs["params"]
        assert params["q"] == "test query"
        assert params["count"] == 5
        assert params["search_lang"] == "en"
        assert params["country"] == "JP"
        assert params["freshness"] == "pw"

    def test_count_clamped_min(self):
        mock_resp = _make_brave_response([])
        with patch("core.tools.web_search.httpx.get", return_value=mock_resp) as mock_get:
            search("test", count=-5)
        params = mock_get.call_args.kwargs["params"]
        assert params["count"] == 1

    def test_count_clamped_max(self):
        mock_resp = _make_brave_response([])
        with patch("core.tools.web_search.httpx.get", return_value=mock_resp) as mock_get:
            search("test", count=100)
        params = mock_get.call_args.kwargs["params"]
        assert params["count"] == 20

    def test_ja_language_mapping(self):
        mock_resp = _make_brave_response([])
        with patch("core.tools.web_search.httpx.get", return_value=mock_resp) as mock_get:
            search("test", lang="ja")
        params = mock_get.call_args.kwargs["params"]
        assert params["search_lang"] == "jp"

    def test_freshness_not_included_when_none(self):
        mock_resp = _make_brave_response([])
        with patch("core.tools.web_search.httpx.get", return_value=mock_resp) as mock_get:
            search("test", freshness=None)
        params = mock_get.call_args.kwargs["params"]
        assert "freshness" not in params

    def test_missing_api_key_raises(self):
        with patch("core.tools.web_search.get_credential", side_effect=ToolConfigError("no key")):
            with pytest.raises(ToolConfigError):
                search("test")

    def test_http_error_propagated(self):
        error_resp = httpx.Response(
            500,
            text="Internal Server Error",
            request=httpx.Request("GET", BRAVE_SEARCH_URL),
        )
        with patch("core.tools.web_search.httpx.get", return_value=error_resp):
            with pytest.raises(httpx.HTTPStatusError):
                search("test")

    def test_empty_results(self):
        data = {"web": {"results": []}}
        mock_resp = httpx.Response(200, json=data, request=httpx.Request("GET", BRAVE_SEARCH_URL))
        with patch("core.tools.web_search.httpx.get", return_value=mock_resp):
            results = search("obscure query")
        assert results == []

    def test_auth_header_set(self):
        mock_resp = _make_brave_response([])
        with patch("core.tools.web_search.httpx.get", return_value=mock_resp) as mock_get:
            search("test")
        headers = mock_get.call_args.kwargs["headers"]
        assert headers["X-Subscription-Token"] == "test-brave-key-123"


# ── _strip_html ───────────────────────────────────────────────────


class TestStripHtml:
    def test_strips_tags(self):
        assert _strip_html("<b>bold</b> text") == "bold text"

    def test_unescapes_entities(self):
        assert _strip_html("a &amp; b &lt; c") == "a & b < c"

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_nested_tags(self):
        assert _strip_html("<div><p>hello</p></div>") == "hello"


# ── format_results ────────────────────────────────────────────────


class TestFormatResults:
    def test_no_results(self):
        assert format_results([]) == "No results found."

    def test_formats_results(self):
        results = [
            {"title": "Test Title", "url": "https://test.com", "description": "A test desc"},
        ]
        output = format_results(results)
        assert "1. Test Title" in output
        assert "https://test.com" in output
        assert "A test desc" in output

    def test_html_stripped_from_description(self):
        results = [
            {"title": "T", "url": "U", "description": "<em>highlighted</em> text"},
        ]
        output = format_results(results)
        assert "<em>" not in output
        assert "highlighted text" in output


# ── get_tool_schemas ──────────────────────────────────────────────


class TestDispatch:
    """Dispatch function parameter mapping tests."""

    @patch("core.tools.web_search.search")
    def test_limit_mapped_to_count(self, mock_search):
        """'limit' in args should be mapped to 'count' for search()."""
        mock_search.return_value = [{"title": "T", "url": "U", "description": "D"}]
        dispatch("web_search", {"query": "test", "limit": 3})
        mock_search.assert_called_once_with(query="test", count=3)

    @patch("core.tools.web_search.search")
    def test_count_passed_directly(self, mock_search):
        """'count' in args should be passed through without mapping."""
        mock_search.return_value = []
        dispatch("web_search", {"query": "test", "count": 7})
        mock_search.assert_called_once_with(query="test", count=7)

    @patch("core.tools.web_search.search")
    def test_anima_dir_stripped(self, mock_search):
        """anima_dir should be stripped before calling search()."""
        mock_search.return_value = []
        dispatch("web_search", {"query": "test", "anima_dir": "/tmp/a"})
        mock_search.assert_called_once_with(query="test")

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch("nonexistent", {})


class TestGetToolSchemas:
    def test_returns_empty_list(self):
        """External tool modules return empty schema lists (schemas from dispatch layer)."""
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)
        assert schemas == []
