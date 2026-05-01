from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# LoCoMo benchmark: Neo4j backend adapter using MemoryBackend interface
import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchmarks.locomo.adapter import (
    _resolve_llm_kwargs,
    _session_indices,
)

try:
    from dateutil import parser as dateutil_parser
except ImportError:
    dateutil_parser = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ── Constants ──────────

_NEO4J_URI = "bolt://localhost:7687"
_NEO4J_USER = "neo4j"
_NEO4J_PASSWORD = "animaworks"
_NEO4J_DATABASE = "neo4j"

_ANSWER_SYSTEM = (
    "You are an expert assistant answering questions about past conversations "
    "based on the provided context."
)

_ANSWER_TEMPLATE = """# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to timestamps (event_time) to determine when events occurred
3. If memories contain contradictory information, prioritize the most recent memory
4. Always convert relative time references (yesterday, last week, next month) to specific dates using the event_time timestamps
5. Timestamps represent the actual time the event occurred, not when it was mentioned
6. When multiple items are asked for, answer with a comma-separated list of short phrases
7. If the context supports a reasonable inference, provide your best answer — only say "No information available." when the context has absolutely no relevant information
8. Keep your answer as brief and direct as possible — a short phrase or sentence

Example:
Memory: (event_time: 2023-05-08T13:56:00) I went to the vet yesterday.
Question: When did I go to the vet?
Answer: 7 May 2023

Context:
{context}

Question: {question}
Answer:"""


# ── Ablation flags ──────────


class AblationFlags:
    """Control which Neo4j features are active during benchmark."""

    def __init__(
        self,
        *,
        reranker: bool = True,
        bfs: bool = True,
        community: bool = True,
        invalidation: bool = True,
    ) -> None:
        self.reranker = reranker
        self.bfs = bfs
        self.community = community
        self.invalidation = invalidation

    def label(self) -> str:
        disabled = []
        if not self.reranker:
            disabled.append("no_reranker")
        if not self.bfs:
            disabled.append("no_bfs")
        if not self.community:
            disabled.append("no_community")
        if not self.invalidation:
            disabled.append("no_invalidation")
        return "_".join(disabled) if disabled else "full"

    def __repr__(self) -> str:
        return (
            f"AblationFlags(reranker={self.reranker}, bfs={self.bfs}, "
            f"community={self.community}, invalidation={self.invalidation})"
        )


# ── Neo4j Adapter ──────────


class Neo4jLoCoMoAdapter:
    """Neo4j-backed LoCoMo adapter using the MemoryBackend interface.

    Manages its own group_id namespace so benchmark data doesn't
    contaminate production Anima memories.
    """

    def __init__(
        self,
        *,
        top_k: int = 10,
        ablation: AblationFlags | None = None,
        answer_model: str = "openai/qwen3.6-35b-a3b",
    ) -> None:
        self._top_k = top_k
        self._ablation = ablation or AblationFlags()
        self._answer_model = answer_model
        self._group_id = f"locomo_bench_{uuid4().hex[:8]}"
        self._backend: Any = None
        # Persistent event loop in a background thread for Neo4j driver
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize Neo4jGraphBackend with benchmark group_id."""
        import json
        import tempfile

        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        self._tmp_anima_dir = Path(tempfile.mkdtemp(prefix="locomo-neo4j-"))
        (self._tmp_anima_dir / "episodes").mkdir(parents=True, exist_ok=True)

        # Write status.json so extraction pipeline can resolve model + api_base
        llm_kwargs = _resolve_llm_kwargs(self._answer_model)
        status: dict[str, Any] = {
            "extraction_model": self._answer_model,
        }
        if llm_kwargs.get("api_base"):
            status["extraction_api_base"] = llm_kwargs["api_base"]
        if llm_kwargs.get("api_key"):
            status["extraction_api_key"] = llm_kwargs["api_key"]
        if llm_kwargs.get("extra_body"):
            status["extraction_extra_body"] = llm_kwargs["extra_body"]
        status["extraction_timeout"] = 120
        (self._tmp_anima_dir / "status.json").write_text(
            json.dumps(status), encoding="utf-8",
        )

        self._backend = Neo4jGraphBackend(
            self._tmp_anima_dir,
            uri=_NEO4J_URI,
            user=_NEO4J_USER,
            password=_NEO4J_PASSWORD,
            database=_NEO4J_DATABASE,
            group_id=self._group_id,
        )

        if not self._ablation.invalidation:
            self._backend._get_invalidator = lambda: _NoOpInvalidator()

    def _run(self, coro: Any) -> Any:
        """Run async coroutine on the persistent event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=300)

    def reset(self) -> None:
        """Clear all benchmark data from Neo4j for this group_id."""
        self._run(self._backend.reset())
        if hasattr(self._backend, "clear_resolver_cache"):
            self._backend.clear_resolver_cache()

    def ingest_conversation(self, sample: dict[str, Any]) -> int:
        """Ingest one LoCoMo conversation into Neo4j as episodes."""
        sample_id = sample.get("sample_id", "unknown")
        conv = sample.get("conversation")
        if not isinstance(conv, dict):
            raise TypeError("sample['conversation'] must be a dict")

        indices = _session_indices(conv)
        total_chunks = 0

        for n in indices:
            sk = f"session_{n}"
            dk = f"session_{n}_date_time"
            turns = conv.get(sk)
            when = str(conv.get(dk, "")).strip() or None

            if not isinstance(turns, list) or not turns:
                continue

            lines: list[str] = []
            for turn in turns:
                if isinstance(turn, dict):
                    sp = str(turn.get("speaker", "?"))
                    txt = (turn.get("text") or "").strip()
                    cap = turn.get("blip_caption")
                    if cap:
                        txt += f" [image: {cap}]"
                    if not txt and turn.get("query"):
                        txt = f"(image search: {turn.get('query')})"
                    lines.append(f"{sp}: {txt}")
                elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
                    lines.append(f"{turn[0]}: {turn[1]}")

            session_text = "\n".join(lines)
            if not session_text.strip():
                continue

            metadata: dict[str, Any] = {
                "episode_uuid": f"{self._group_id}_s{sample_id}_sess{n}",
                "description": f"Conversation {sample_id}, Session {n}",
            }
            if when:
                iso_when = _parse_datetime_to_iso(when)
                if iso_when:
                    metadata["valid_at"] = iso_when

            chunks = self._run(
                self._backend.ingest_text(session_text, source=f"conv-{sample_id}_session_{n}", metadata=metadata)
            )
            total_chunks += chunks

        if hasattr(self._backend, "clear_resolver_cache"):
            self._backend.clear_resolver_cache()

        return total_chunks

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Retrieve relevant memories from Neo4j."""
        return self._run(self._retrieve_async(question))

    async def _retrieve_async(self, question: str) -> list[dict[str, Any]]:
        """Hybrid retrieval with ablation-aware patching.

        Searches both facts and episodes for comprehensive coverage.
        """
        from core.memory.graph.search import HybridSearch

        driver = await self._backend._ensure_driver()
        search = HybridSearch(driver, self._group_id)

        from core.memory.rag.singleton import generate_embeddings

        query_embedding = await asyncio.to_thread(generate_embeddings, [question])
        qe = query_embedding[0] if query_embedding else []

        original_bfs = search._bfs_search
        original_rerank = None

        if not self._ablation.bfs:
            search._bfs_search = _noop_search

        if not self._ablation.reranker:
            original_rerank = search.search
            search.search = self._search_no_reranker(search)

        try:
            # Search facts (entity relationships)
            fact_results = await search.search(
                question,
                scope="fact",
                limit=self._top_k * 2,
                query_embedding=qe,
            )
            # Search episodes (raw conversation content)
            episode_results = await search.search(
                question,
                scope="episode",
                limit=self._top_k,
                query_embedding=qe,
            )
        finally:
            search._bfs_search = original_bfs
            if original_rerank:
                search.search = original_rerank

        out: list[dict[str, Any]] = []
        seen_content: set[str] = set()

        # Episode content (conversation text — higher priority)
        for r in episode_results:
            content = (r.get("content") or "").strip()
            if content and content not in seen_content:
                seen_content.add(content)
                valid_at = r.get("valid_at") or ""
                out.append({
                    "content": content,
                    "score": float(r.get("ce_score", r.get("rrf_score", r.get("score", 0.0)))),
                    "valid_at": valid_at,
                    "metadata": {"scope": "episode"},
                })

        # Fact content (structured knowledge)
        for r in fact_results:
            src = r.get("source_name", "")
            tgt = r.get("target_name", "")
            fact_text = r.get("fact", "")
            if src and tgt and fact_text:
                content = f"{src} → {tgt}: {fact_text}"
            else:
                content = fact_text or r.get("content", "") or r.get("name", "")
            content = content.strip()
            if content and content not in seen_content:
                seen_content.add(content)
                valid_at = r.get("valid_at") or ""
                out.append({
                    "content": content,
                    "score": float(r.get("ce_score", r.get("rrf_score", r.get("score", 0.0)))),
                    "valid_at": valid_at,
                    "metadata": {"scope": "fact"},
                })

        return out[: self._top_k]

    def _search_no_reranker(self, search: Any):
        """Create a patched search method that skips cross-encoder reranking."""

        async def patched_search(
            query: str,
            *,
            scope: str = "fact",
            limit: int = 10,
            as_of_time: str | None = None,
            time_start: str | None = None,
            time_end: str | None = None,
            query_embedding: list[float] | None = None,
            edge_type_filter: str | None = None,
        ) -> list[dict]:
            from datetime import UTC, datetime

            if not query or not query.strip():
                raise ValueError("Search query must not be empty")
            if as_of_time is None:
                as_of_time = datetime.now(tz=UTC).isoformat()

            results = await asyncio.gather(
                search._vector_search(
                    query, scope, as_of_time, query_embedding,
                    time_start=time_start, time_end=time_end,
                ),
                search._fulltext_search(
                    query, scope, as_of_time,
                    time_start=time_start, time_end=time_end,
                ),
                search._bfs_search(
                    query, scope, as_of_time, query_embedding,
                    time_start=time_start, time_end=time_end,
                ),
                return_exceptions=True,
            )

            result_lists: list[list[dict]] = []
            for r in results:
                if isinstance(r, Exception):
                    continue
                if r:
                    result_lists.append(r)

            if not result_lists:
                return []

            from core.memory.graph.rrf import rrf_merge

            merged = rrf_merge(result_lists, top_k=min(30, limit * 3), k=search._rrf_k)
            return merged[:limit]

        return patched_search

    def answer(self, question: str, context: list[dict], *, model: str | None = None) -> str:
        """Generate answer using LLM."""
        use_model = model or self._answer_model
        parts = []
        for i, c in enumerate(context, start=1):
            t = (c.get("content") or "").strip()
            if t:
                va = c.get("valid_at") or ""
                if va:
                    parts.append(f"[{i}] (event_time: {va}) {t}")
                else:
                    parts.append(f"[{i}] {t}")
        ctx_joined = "\n\n".join(parts)
        user_prompt = _ANSWER_TEMPLATE.format(context=ctx_joined, question=question)
        return _llm_complete(use_model, user_prompt, system=_ANSWER_SYSTEM)

    def cleanup(self) -> None:
        """Clean up Neo4j data and temp dir."""
        try:
            self._run(self._backend.reset())
        except Exception:
            logger.warning("Neo4j cleanup (reset) failed", exc_info=True)
        try:
            self._run(self._backend.close())
        except Exception:
            logger.debug("Neo4j driver close failed", exc_info=True)

        # Stop the persistent event loop
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)
        self._loop.close()

        import shutil

        if self._tmp_anima_dir and self._tmp_anima_dir.exists():
            shutil.rmtree(self._tmp_anima_dir, ignore_errors=True)

    def __enter__(self) -> Neo4jLoCoMoAdapter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.cleanup()


# ── Helpers ──────────


def _llm_complete(model: str, user_prompt: str, *, system: str | None = None) -> str:
    """LiteLLM completion with auto-resolved base_url for local models."""
    import litellm

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_prompt})
    extra = _resolve_llm_kwargs(model)
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            r = litellm.completion(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                **extra,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            logger.warning("_llm_complete attempt %s/3 failed: %s", attempt, e)
            if attempt < 3:
                time.sleep(0.5 * (2 ** (attempt - 1)))
    assert last_err is not None
    raise last_err


def _parse_datetime_to_iso(text: str) -> str | None:
    """Try to parse a human-readable date string to ISO format."""
    if not text:
        return None
    if "T" in text and ("-" in text[:10]):
        return text
    if dateutil_parser is not None:
        try:
            dt = dateutil_parser.parse(text)
            return dt.isoformat()
        except (ValueError, OverflowError):
            return None
    return None


class _NoOpInvalidator:
    """Stub invalidator that does nothing (for ablation)."""

    async def find_and_invalidate(self, **kwargs: Any) -> list:
        return []


async def _noop_search(*args: Any, **kwargs: Any) -> list[dict]:
    """No-op search source for ablation."""
    return []
