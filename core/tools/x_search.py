# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""X (Twitter) Search tool for AnimaWorks.

Migrated from ~/bin/x-search.
Uses httpx instead of urllib.request.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from core.tools._base import ToolConfigError, get_credential, logger

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "search": {"expected_seconds": 30, "background_eligible": False},
}


# ---------------------------------------------------------------------------
# X API v2 client
# ---------------------------------------------------------------------------

class XSearchClient:
    """X (Twitter) API v2 client."""

    BASE_URL = "https://api.twitter.com/2"

    def __init__(self, bearer_token: str | None = None) -> None:
        self.bearer_token = bearer_token or get_credential(
            "x_twitter", "x_search", env_var="TWITTER_BEARER_TOKEN"
        )

    # -- internal helpers ---------------------------------------------------

    def _request(self, endpoint: str, params: dict[str, Any]) -> dict:
        """Execute an authenticated API request."""
        url = f"{self.BASE_URL}/{endpoint}"

        response = httpx.get(
            url,
            params=params,
            headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "User-Agent": "AnimaWorks/1.0",
            },
            timeout=30.0,
        )

        if response.status_code == 429:
            raise RuntimeError("Rate limit exceeded. Try again later.")
        if response.status_code == 401:
            raise RuntimeError("Invalid bearer token.")
        if response.status_code == 403:
            raise RuntimeError(f"Access forbidden: {response.text}")

        response.raise_for_status()
        return response.json()

    def _format_tweets(self, result: dict) -> list[dict]:
        """Normalize API response into a flat list of tweet dicts."""
        tweets: list[dict] = []
        data = result.get("data", [])
        users = {
            u["id"]: u
            for u in result.get("includes", {}).get("users", [])
        }

        for tweet in data:
            author_id = tweet.get("author_id", "")
            author = users.get(author_id, {})
            metrics = tweet.get("public_metrics", {})

            tweets.append({
                "id": tweet.get("id"),
                "text": tweet.get("text", ""),
                "created_at": tweet.get("created_at", ""),
                "username": author.get("username", ""),
                "author_name": author.get("name", ""),
                "verified": author.get("verified", False),
                "likes": metrics.get("like_count", 0),
                "retweets": metrics.get("retweet_count", 0),
                "replies": metrics.get("reply_count", 0),
                "impressions": metrics.get("impression_count", 0),
            })

        return tweets

    # -- public API ---------------------------------------------------------

    def search_recent(
        self,
        query: str,
        max_results: int = 10,
        days: int = 7,
    ) -> list[dict]:
        """Search recent tweets.

        Args:
            query: Search query (X search syntax supported).
            max_results: Maximum number of tweets (10-100).
            days: How many days back to search (default 7).

        Returns:
            List of tweet dicts.
        """
        start_time = datetime.now(timezone.utc) - timedelta(days=days)

        params: dict[str, Any] = {
            "query": query,
            "max_results": min(max(10, max_results), 100),
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tweet.fields": "created_at,public_metrics,author_id,text",
            "expansions": "author_id",
            "user.fields": "username,name,verified",
        }

        result = self._request("tweets/search/recent", params)
        return self._format_tweets(result)

    def get_user_tweets(
        self,
        username: str,
        max_results: int = 10,
        days: int | None = None,
    ) -> list[dict]:
        """Get tweets from a specific user.

        Args:
            username: X username (without @).
            max_results: Maximum number of tweets.
            days: Limit to tweets from the last N days (None = no limit).

        Returns:
            List of tweet dicts.
        """
        # Resolve username to user ID
        user_result = self._request(f"users/by/username/{username}", {})
        if "data" not in user_result:
            raise RuntimeError(f"User @{username} not found")

        user_id = user_result["data"]["id"]

        # Calculate optional time window
        start_time: datetime | None = None
        if days is not None:
            start_time = datetime.now(timezone.utc) - timedelta(days=days)

        # Paginate through results
        all_tweets: list[dict] = []
        next_token: str | None = None

        while len(all_tweets) < max_results:
            remaining = max_results - len(all_tweets)
            fetch_count = max(5, min(100, remaining))  # API minimum is 5

            params: dict[str, Any] = {
                "max_results": fetch_count,
                "tweet.fields": "created_at,public_metrics,text,referenced_tweets",
                "exclude": "replies",
            }
            if next_token:
                params["pagination_token"] = next_token
            if start_time:
                params["start_time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

            result = self._request(f"users/{user_id}/tweets", params)
            new_tweets = self._format_tweets(result)

            if not new_tweets:
                break

            all_tweets.extend(new_tweets)

            meta = result.get("meta", {})
            next_token = meta.get("next_token")
            if not next_token:
                break

        # Attach user info to all tweets
        for tweet in all_tweets:
            tweet["username"] = username
            tweet["author_name"] = user_result["data"].get("name", username)

        return all_tweets[:max_results]


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def _format_tweet_text(tweet: dict, verbose: bool = False) -> str:
    """Format a single tweet as human-readable text."""
    lines: list[str] = []

    verified = " [verified]" if tweet.get("verified") else ""
    lines.append(f"@{tweet['username']}{verified} - {tweet['created_at'][:16]}")

    text = tweet["text"].replace("\n", "\n  ")
    lines.append(f"  {text}")

    if verbose:
        lines.append(
            f"  [Likes: {tweet['likes']:,} | "
            f"RTs: {tweet['retweets']:,} | "
            f"Replies: {tweet['replies']:,}]"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anthropic tool_use schemas
# ---------------------------------------------------------------------------

def get_tool_schemas() -> list[dict]:
    """Return Anthropic tool_use schemas for X search tools."""
    return [
        {
            "name": "x_search",
            "description": (
                "Search recent tweets on X (Twitter). "
                "Returns tweets with text, author info, and engagement metrics."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (X search syntax supported).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of tweets to return (10-100, default 10).",
                        "default": 10,
                    },
                    "days": {
                        "type": "integer",
                        "description": "Search tweets from last N days (default 7).",
                        "default": 7,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "x_user_tweets",
            "description": (
                "Get recent tweets from a specific X (Twitter) user. "
                "Returns tweets with text and engagement metrics."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "X username (without @).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of tweets to return (default 10).",
                        "default": 10,
                    },
                    "days": {
                        "type": "integer",
                        "description": "Limit to tweets from last N days (omit for no limit).",
                    },
                },
                "required": ["username"],
            },
        },
    ]


# ── Dispatch ──────────────────────────────────────────

def dispatch(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call by schema name."""
    if name == "x_search":
        client = XSearchClient()
        return client.search_recent(
            query=args["query"],
            max_results=args.get("max_results", 10),
            days=args.get("days", 7),
        )
    if name == "x_user_tweets":
        client = XSearchClient()
        return client.get_user_tweets(
            username=args["username"],
            max_results=args.get("max_results", 10),
            days=args.get("days"),
        )
    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli_main(argv: list[str] | None = None) -> None:
    """Thin CLI entry point for x_search.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        description="Search X (Twitter) using API v2",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query (X search syntax supported)",
    )
    parser.add_argument(
        "-u", "--user",
        type=str,
        help="Get tweets from a specific user (without @)",
    )
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=10,
        help="Number of tweets (default: 10)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Search tweets from last N days (default: 7)",
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show tweet metrics",
    )

    args = parser.parse_args(argv)

    if not args.query and not args.user:
        parser.error("Either query or --user is required")

    client = XSearchClient()

    if args.user:
        tweets = client.get_user_tweets(
            username=args.user,
            max_results=args.count,
            days=args.days if args.days != 7 else None,
        )
        title = f"Tweets from @{args.user}"
    else:
        tweets = client.search_recent(
            query=args.query,
            max_results=args.count,
            days=args.days,
        )
        title = f"Search results for: {args.query}"

    if args.json:
        print(json.dumps(tweets, indent=2, ensure_ascii=False))
    else:
        print("=" * 60)
        print(title)
        print(f"Found {len(tweets)} tweets")
        print("=" * 60)
        print()

        for i, tweet in enumerate(tweets, 1):
            print(f"[{i}] {_format_tweet_text(tweet, args.verbose)}")
            print()