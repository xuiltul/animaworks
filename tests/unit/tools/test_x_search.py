"""Tests for core/tools/x_search.py — X/Twitter search tool."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.tools._base import ToolConfigError
from core.tools.x_search import (
    XSearchClient,
    _format_tweet_text,
    get_tool_schemas,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _set_twitter_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TWITTER_BEARER_TOKEN", "test-bearer-token")


def _make_x_response(
    tweets: list[dict] | None = None,
    users: list[dict] | None = None,
    meta: dict | None = None,
) -> httpx.Response:
    """Build a mock httpx.Response for X API."""
    if tweets is None:
        tweets = [
            {
                "id": "1",
                "text": "Hello world",
                "created_at": "2026-01-01T12:00:00Z",
                "author_id": "u1",
                "public_metrics": {"like_count": 10, "retweet_count": 5, "reply_count": 2, "impression_count": 100},
            },
        ]
    if users is None:
        users = [
            {"id": "u1", "username": "testuser", "name": "Test User", "verified": False},
        ]
    data = {"data": tweets, "includes": {"users": users}}
    if meta:
        data["meta"] = meta
    return httpx.Response(200, json=data, request=httpx.Request("GET", "https://api.twitter.com/2/test"))


# ── XSearchClient ─────────────────────────────────────────────────


class TestXSearchClient:
    def test_init_with_token(self):
        client = XSearchClient(bearer_token="my-token")
        assert client.bearer_token == "my-token"

    def test_init_from_env(self):
        client = XSearchClient()
        assert client.bearer_token == "test-bearer-token"

    def test_init_missing_token(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
        with pytest.raises(ToolConfigError):
            XSearchClient()


class TestXSearchClientRequest:
    def test_request_sets_auth_header(self):
        client = XSearchClient(bearer_token="tok123")
        mock_resp = httpx.Response(
            200, json={"data": []},
            request=httpx.Request("GET", "https://api.twitter.com/2/test"),
        )
        with patch("core.tools.x_search.httpx.get", return_value=mock_resp) as mock_get:
            client._request("test", {})
        headers = mock_get.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer tok123"

    def test_rate_limit_error(self):
        client = XSearchClient(bearer_token="tok")
        resp = httpx.Response(
            429, text="Rate limited",
            request=httpx.Request("GET", "https://api.twitter.com/2/test"),
        )
        with patch("core.tools.x_search.httpx.get", return_value=resp):
            with pytest.raises(RuntimeError, match="Rate limit"):
                client._request("test", {})

    def test_unauthorized_error(self):
        client = XSearchClient(bearer_token="bad")
        resp = httpx.Response(
            401, text="Unauthorized",
            request=httpx.Request("GET", "https://api.twitter.com/2/test"),
        )
        with patch("core.tools.x_search.httpx.get", return_value=resp):
            with pytest.raises(RuntimeError, match="Invalid bearer token"):
                client._request("test", {})

    def test_forbidden_error(self):
        client = XSearchClient(bearer_token="tok")
        resp = httpx.Response(
            403, text="Forbidden",
            request=httpx.Request("GET", "https://api.twitter.com/2/test"),
        )
        with patch("core.tools.x_search.httpx.get", return_value=resp):
            with pytest.raises(RuntimeError, match="Access forbidden"):
                client._request("test", {})


class TestSearchRecent:
    def test_search_recent_success(self):
        client = XSearchClient(bearer_token="tok")
        mock_resp = _make_x_response()
        with patch("core.tools.x_search.httpx.get", return_value=mock_resp):
            tweets = client.search_recent("hello")
        assert len(tweets) == 1
        assert tweets[0]["text"] == "Hello world"
        assert tweets[0]["username"] == "testuser"
        assert tweets[0]["likes"] == 10

    def test_search_recent_clamps_max_results(self):
        client = XSearchClient(bearer_token="tok")
        mock_resp = _make_x_response([])
        with patch("core.tools.x_search.httpx.get", return_value=mock_resp) as mock_get:
            client.search_recent("q", max_results=5)
        params = mock_get.call_args.kwargs["params"]
        assert params["max_results"] == 10  # minimum is 10

    def test_search_recent_sets_start_time(self):
        client = XSearchClient(bearer_token="tok")
        mock_resp = _make_x_response([])
        with patch("core.tools.x_search.httpx.get", return_value=mock_resp) as mock_get:
            client.search_recent("q", days=3)
        params = mock_get.call_args.kwargs["params"]
        assert "start_time" in params


class TestGetUserTweets:
    def test_get_user_tweets_success(self):
        client = XSearchClient(bearer_token="tok")
        user_resp = httpx.Response(
            200,
            json={"data": {"id": "u1", "name": "Test User"}},
            request=httpx.Request("GET", "https://api.twitter.com/2/users/by/username/testuser"),
        )
        tweets_resp = _make_x_response(meta={})
        responses = [user_resp, tweets_resp]
        with patch("core.tools.x_search.httpx.get", side_effect=responses):
            tweets = client.get_user_tweets("testuser", max_results=1)
        assert len(tweets) == 1
        assert tweets[0]["username"] == "testuser"

    def test_user_not_found(self):
        client = XSearchClient(bearer_token="tok")
        user_resp = httpx.Response(
            200,
            json={"errors": [{"detail": "not found"}]},
            request=httpx.Request("GET", "https://api.twitter.com/2/users/by/username/nobody"),
        )
        with patch("core.tools.x_search.httpx.get", return_value=user_resp):
            with pytest.raises(RuntimeError, match="not found"):
                client.get_user_tweets("nobody")


class TestFormatTweets:
    def test_format_tweets(self):
        client = XSearchClient(bearer_token="tok")
        result = {
            "data": [
                {
                    "id": "1",
                    "text": "test tweet",
                    "created_at": "2026-01-01T00:00:00Z",
                    "author_id": "u1",
                    "public_metrics": {"like_count": 5, "retweet_count": 3, "reply_count": 1, "impression_count": 50},
                },
            ],
            "includes": {
                "users": [
                    {"id": "u1", "username": "alice", "name": "Alice", "verified": True},
                ],
            },
        }
        tweets = client._format_tweets(result)
        assert len(tweets) == 1
        assert tweets[0]["username"] == "alice"
        assert tweets[0]["verified"] is True
        assert tweets[0]["likes"] == 5

    def test_format_tweets_empty(self):
        client = XSearchClient(bearer_token="tok")
        result = {"data": [], "includes": {"users": []}}
        assert client._format_tweets(result) == []

    def test_format_tweets_no_user_match(self):
        client = XSearchClient(bearer_token="tok")
        result = {
            "data": [{"id": "1", "text": "t", "created_at": "", "author_id": "unknown"}],
            "includes": {"users": []},
        }
        tweets = client._format_tweets(result)
        assert tweets[0]["username"] == ""


# ── _format_tweet_text ────────────────────────────────────────────


class TestFormatTweetText:
    def test_basic_format(self):
        tweet = {
            "username": "bob",
            "verified": False,
            "created_at": "2026-01-15T10:30:00Z",
            "text": "Hello World",
            "likes": 10,
            "retweets": 5,
            "replies": 2,
        }
        output = _format_tweet_text(tweet)
        assert "@bob" in output
        assert "Hello World" in output
        assert "[verified]" not in output

    def test_verified_badge(self):
        tweet = {
            "username": "celeb",
            "verified": True,
            "created_at": "2026-01-15T10:30:00Z",
            "text": "Hi",
            "likes": 0,
            "retweets": 0,
            "replies": 0,
        }
        output = _format_tweet_text(tweet)
        assert "[verified]" in output

    def test_verbose_mode(self):
        tweet = {
            "username": "u",
            "verified": False,
            "created_at": "2026-01-15T10:30:00Z",
            "text": "test",
            "likes": 1000,
            "retweets": 500,
            "replies": 100,
        }
        output = _format_tweet_text(tweet, verbose=True)
        assert "Likes:" in output
        assert "RTs:" in output
        assert "Replies:" in output


# ── get_tool_schemas ──────────────────────────────────────────────


class TestGetToolSchemas:
    def test_returns_two_schemas(self):
        schemas = get_tool_schemas()
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert names == {"x_search", "x_user_tweets"}

    def test_x_search_schema(self):
        schema = [s for s in get_tool_schemas() if s["name"] == "x_search"][0]
        assert "query" in schema["input_schema"]["required"]

    def test_x_user_tweets_schema(self):
        schema = [s for s in get_tool_schemas() if s["name"] == "x_user_tweets"][0]
        assert "username" in schema["input_schema"]["required"]
