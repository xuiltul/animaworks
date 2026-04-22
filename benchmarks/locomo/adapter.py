from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# LoCoMo benchmark: isolated AnimaWorks RAG ↔ dataset adapter
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Project root (development / `python -m` from repo root)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ── Constants ──────────

ANIMA_NAME = "locomo_bench"
SEARCH_MODES: tuple[str, ...] = ("vector", "vector_graph", "scope_all")
_SESSION_RE = re.compile(r"^session_(\d+)$")

# ── Dependency checks ──────────


def _ensure_rag_stack() -> None:
    """Verify optional RAG stack imports; raise with actionable errors."""
    try:
        import chromadb  # noqa: F401, PLC0415 - runtime check
    except ImportError as e:
        raise ImportError(
            "LoCoMo adapter requires `chromadb`. Install the AnimaWorks RAG extra (e.g. `pip install chromadb`).",
        ) from e
    try:
        import sentence_transformers  # noqa: F401, PLC0415
    except ImportError as e:
        raise ImportError(
            "LoCoMo adapter requires `sentence-transformers`. "
            "Install the AnimaWorks RAG extra (e.g. `pip install sentence-transformers`).",
        ) from e
    try:
        import rank_bm25  # noqa: F401, PLC0415
    except ImportError as e:
        raise ImportError(
            "LoCoMo adapter requires `rank-bm25` for `scope_all` (BM25+RRF).",
        ) from e


# Defer to core (same tokenizer rules as activity BM25) when available
try:
    from core.memory.bm25 import tokenize as _bm25_tokenize
except ImportError:

    def _bm25_tokenize(text: str) -> list[str]:
        """Simple word token fallback if core not on path."""
        return re.findall(r"[\w]+", text, flags=re.UNICODE)[:2000]

# ── load_dataset ──────────


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load LoCoMo JSON (e.g. ``locomo10.json``) and normalize category-5 QAs.

    For adversarial (category 5) items, copies ``adversarial_answer`` to
    ``answer`` when ``answer`` is missing.

    Args:
        path: Path to the dataset file.

    Returns:
        List of sample dicts (``sample_id``, ``conversation``, ``qa``, etc.).
    """
    with path.open(encoding="utf-8") as f:
        data: Any = json.load(f)
    if not isinstance(data, list):
        raise ValueError("LoCoMo dataset top-level value must be a list")
    for sample in data:
        if not isinstance(sample, dict):
            continue
        qa_list = sample.get("qa")
        if not isinstance(qa_list, list):
            continue
        for item in qa_list:
            if not isinstance(item, dict):
                continue
            try:
                cat = int(item.get("category", 0) or 0)
            except (TypeError, ValueError):
                continue
            if cat != 5:
                continue
            if "answer" not in item and "adversarial_answer" in item:
                item["answer"] = item["adversarial_answer"]
    return data  # type: ignore[return-value]


# ── Markdown helpers ──────────


def _session_indices(conversation: dict[str, Any]) -> list[int]:
    """Return sorted session numbers present as ``session_N`` keys."""
    indices: set[int] = set()
    for k in conversation:
        m = _SESSION_RE.match(k)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def _format_turn(turn: Any) -> str:
    """Format one dialog turn to a markdown line."""
    if isinstance(turn, dict):
        sp = str(turn.get("speaker", "?"))
        txt = (turn.get("text") or "").strip()
        extras: list[str] = []
        cap = turn.get("blip_caption")
        if cap:
            extras.append(f" [image: {cap}]")
        if not txt and turn.get("query"):
            txt = f"(image search: {turn.get('query')})"
        line = f"**{sp}**: {txt}{''.join(extras)}"
        return line
    if isinstance(turn, (list, tuple)) and len(turn) >= 2:
        return f"**{turn[0]}**: {turn[1]}"
    return ""


def _build_episode_markdown(
    sample_id: str,
    conversation: dict[str, Any],
) -> str:
    """Build a single markdown document from a LoCoMo ``conversation`` object."""
    sp_a = str(conversation.get("speaker_a", "Speaker A"))
    sp_b = str(conversation.get("speaker_b", "Speaker B"))
    header = f"<!-- sample_id: {sample_id} | speakers: {sp_a}, {sp_b} -->\n\n"
    parts: list[str] = [header]
    for n in _session_indices(conversation):
        sk = f"session_{n}"
        dk = f"session_{n}_date_time"
        when = str(conversation.get(dk, "")).strip() or "unknown date"
        turns = conversation.get(sk)
        parts.append(f"## Session {n} — {when}\n")
        if isinstance(turns, list):
            for turn in turns:
                line = _format_turn(turn)
                if line:
                    parts.append(line)
                    parts.append("")
        else:
            parts.append("(no turns)\n")
        parts.append("\n")
    return "".join(parts).strip() + "\n"


def _episode_stem_for_sample(sample_id: str | int) -> str:
    """Stable filename stem: ``conv-26`` → ``conv-26.md`` (no ``conv-conv-``)."""
    s = str(sample_id).strip() or "unknown"
    if s.startswith("conv-"):
        return s
    return f"conv-{s}"


# ── Adapter ──────────


class AnimaWorksLoCoMoAdapter:
    """AnimaWorks RAG stack wired for LoCoMo in an isolated data directory.

    Warning:
        NOT thread-safe. Uses ``os.environ["ANIMAWORKS_DATA_DIR"]`` override,
        which is process-global. Only one adapter instance should be active per
        process at a time.  For parallel benchmark runs, use separate processes.
    """

    def __init__(self, search_mode: str = "vector", *, top_k: int = 5) -> None:
        """
        Args:
            search_mode: ``vector`` | ``vector_graph`` | ``scope_all``
            top_k: Number of hits to return from retrieval
        """
        if search_mode not in SEARCH_MODES:
            raise ValueError(f"search_mode must be one of {SEARCH_MODES}, got {search_mode!r}")
        _ensure_rag_stack()
        self._search_mode = search_mode
        self._top_k = top_k
        self._temp_dir: str | None = None
        self._previous_animaworks_data: str | None = None
        self._own_data_env = False
        self._anima_dir: Path | None = None
        self._episodes_dir: Path | None = None
        self._vector_store: Any = None
        self._indexer: Any = None
        self._retriever: Any = None
        self._bm25_corpus: list[tuple[str, dict[str, Any]]] | None = None
        self._bm25_index: Any = None
        # Deferred heavy init
        self._init_isolated_rag()

    def _init_isolated_rag(self) -> None:
        """Create temp data dir, env override, and RAG components."""
        from core.memory.rag.indexer import MemoryIndexer  # noqa: PLC0415
        from core.memory.rag.retriever import MemoryRetriever  # noqa: PLC0415
        from core.memory.rag.singleton import get_embedding_model  # noqa: PLC0415
        from core.memory.rag.store import ChromaVectorStore  # noqa: PLC0415

        self._previous_animaworks_data = os.environ.get("ANIMAWORKS_DATA_DIR")
        real_data_dir = Path(os.environ.get("ANIMAWORKS_DATA_DIR", "~/.animaworks")).expanduser().resolve()
        self._temp_dir = tempfile.mkdtemp(prefix="animaworks-locomo-")
        os.environ["ANIMAWORKS_DATA_DIR"] = self._temp_dir
        self._own_data_env = True
        self._anima_dir = Path(self._temp_dir) / "animas" / ANIMA_NAME
        self._episodes_dir = self._anima_dir / "episodes"
        for sub in (
            "episodes",
            "knowledge",
            "procedures",
            "common_knowledge",
        ):
            (self._anima_dir / sub).mkdir(parents=True, exist_ok=True)
        real_models = real_data_dir / "models"
        tmp_models = Path(self._temp_dir) / "models"
        if real_models.is_dir() and not tmp_models.exists():
            tmp_models.symlink_to(real_models)

        vdir = self._anima_dir / "vectordb"
        vdir.mkdir(parents=True, exist_ok=True)
        try:
            self._vector_store = ChromaVectorStore(persist_dir=vdir)
        except Exception as e:
            logger.error("ChromaVectorStore init failed: %s", e)
            raise
        try:
            emb = get_embedding_model()
        except Exception as e:
            logger.error("Embedding model load failed: %s", e)
            raise
        self._indexer = MemoryIndexer(
            self._vector_store,
            ANIMA_NAME,
            self._anima_dir,
            embedding_model=emb,
        )
        self._retriever = MemoryRetriever(
            self._vector_store,
            self._indexer,
            knowledge_dir=self._anima_dir / "knowledge",
        )

    @property
    def _index_meta_path(self) -> Path:
        from core.memory.rag.indexer import INDEX_META_FILE  # noqa: PLC0415

        assert self._anima_dir is not None
        return self._anima_dir / INDEX_META_FILE

    def reset(self) -> None:
        """Remove indexed vectors for this anima, episode files, and index metadata."""
        assert self._vector_store is not None and self._anima_dir is not None and self._episodes_dir is not None
        for name in self._vector_store.list_collections():
            if name.startswith(ANIMA_NAME):
                self._vector_store.delete_collection(name)
        if self._episodes_dir.exists():
            shutil.rmtree(self._episodes_dir)
        self._episodes_dir.mkdir(parents=True, exist_ok=True)
        self._bm25_corpus = None
        self._bm25_index = None
        meta = self._index_meta_path
        if meta.exists():
            try:
                meta.unlink()
            except OSError as e:
                logger.warning("Failed to remove %s: %s", meta, e)

    def ingest_conversation(self, sample: dict[str, Any]) -> int:
        """Write one sample's ``conversation`` to ``episodes/`` and re-index it.

        Args:
            sample: LoCoMo sample dict with ``sample_id`` and ``conversation``.

        Returns:
            Number of vector chunks written for the episode file.
        """
        assert self._indexer is not None and self._episodes_dir is not None
        sample_id = sample.get("sample_id", "unknown")
        conv = sample.get("conversation")
        if not isinstance(conv, dict):
            raise TypeError("sample['conversation'] must be a dict")
        stem = _episode_stem_for_sample(str(sample_id))
        md = _build_episode_markdown(str(sample_id), conv)
        file_path = self._episodes_dir / f"{stem}.md"
        file_path.write_text(md, encoding="utf-8")
        n = self._indexer.index_file(file_path, memory_type="episodes", force=True)
        self._bm25_corpus = None
        self._bm25_index = None
        return n

    def _retrieval_to_dicts(self, results: list[Any]) -> list[dict[str, Any]]:
        from core.memory.rag.retriever import RetrievalResult  # noqa: PLC0415

        out: list[dict[str, Any]] = []
        for r in results:
            if not isinstance(r, RetrievalResult):
                continue
            meta = r.metadata if isinstance(r.metadata, dict) else {}
            out.append(
                {
                    "content": r.content,
                    "score": float(r.score),
                    "metadata": dict(meta),
                },
            )
        return out

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Run retrieval for ``question`` following ``self._search_mode``."""
        assert self._retriever is not None
        if self._search_mode == "vector":
            res = self._retriever.search(
                query=question,
                anima_name=ANIMA_NAME,
                memory_type="episodes",
                top_k=self._top_k,
                enable_spreading_activation=False,
            )
            return self._retrieval_to_dicts(res)
        if self._search_mode == "vector_graph":
            res = self._retriever.search(
                query=question,
                anima_name=ANIMA_NAME,
                memory_type="episodes",
                top_k=self._top_k,
                enable_spreading_activation=True,
            )
            return self._retrieval_to_dicts(res)
        # scope_all: dense + graph on episodes, BM25 on episode text, RRF merge
        pool = max(self._top_k * 4, 20)
        vec = self._retriever.search(
            query=question,
            anima_name=ANIMA_NAME,
            memory_type="episodes",
            top_k=pool,
            enable_spreading_activation=True,
        )
        vec_dicts = self._retrieval_to_dicts(vec)
        bm25_dicts = self._bm25_search(question, top_k=pool)
        return self._rrf_merge(vec_dicts, bm25_dicts, k=60)[: self._top_k]

    def _build_bm25_cache(self) -> None:
        """Build BM25 index from episode files; cached until reset/ingest."""
        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        assert self._episodes_dir is not None
        documents: list[tuple[str, dict[str, Any]]] = []
        for p in sorted(self._episodes_dir.glob("*.md")):
            raw = p.read_text(encoding="utf-8")
            segs: list[str]
            if re.search(r"(?m)^##\s+Session\s", raw):
                segs = [s.strip() for s in re.split(r"(?m)(?=^##\s+Session\s)", raw) if s.strip()]
            else:
                segs = [raw] if raw.strip() else []
            for j, seg in enumerate(segs):
                documents.append((seg, {"source_file": p.name, "section": j}))
        self._bm25_corpus = documents
        if documents:
            tokenized = [_bm25_tokenize(doc) for doc, _ in documents]
            self._bm25_index = BM25Okapi(tokenized) if any(tokenized) else None
        else:
            self._bm25_index = None

    def _bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """BM25 keyword search using cached index (built after ingest)."""
        if self._bm25_corpus is None:
            self._build_bm25_cache()
        documents = self._bm25_corpus
        if not documents:
            return []
        qtok = _bm25_tokenize(query)
        if not qtok or self._bm25_index is None:
            return [
                {
                    "content": documents[0][0],
                    "score": 0.0,
                    "metadata": {**documents[0][1], "search_method": "bm25_degenerate"},
                }
            ][:1]
        scores = self._bm25_index.get_scores(qtok)
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[: top_k * 2]
        out: list[dict[str, Any]] = []
        for i in order:
            out.append(
                {
                    "content": documents[i][0],
                    "score": float(scores[i]),
                    "metadata": {**documents[i][1], "search_method": "bm25"},
                },
            )
        return out

    def _rrf_merge(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Reciprocal rank fusion: merge two ranked lists, deduplicated by content."""
        scores: dict[str, float] = {}
        by_key: dict[str, dict[str, Any]] = {}

        def _add(lst: list[dict[str, Any]], *, method: str) -> None:
            for rank, item in enumerate(lst, start=1):
                content = (item.get("content") or "").strip()
                if not content:
                    continue
                key = content
                part = 1.0 / (k + rank)
                scores[key] = scores.get(key, 0.0) + part
                if key not in by_key:
                    meta = item.get("metadata")
                    m = dict(meta) if isinstance(meta, dict) else {}
                    m["rrf_method"] = method
                    by_key[key] = {
                        "content": item.get("content", ""),
                        "metadata": m,
                    }

        _add(vector_results, method="vector")
        _add(bm25_results, method="bm25")
        merged_keys = sorted(by_key, key=lambda x: scores.get(x, 0.0), reverse=True)
        merged: list[dict[str, Any]] = []
        for key in merged_keys:
            row = dict(by_key[key])
            row["score"] = scores[key]
            row["metadata"] = dict(row.get("metadata", {}))
            row["metadata"]["search_method"] = "rrf"
            merged.append(row)
        return merged

    def _complete_sync(self, messages: list[dict[str, str]], model: str) -> str:
        """Synchronous LLM call with retries (used inside running event loop)."""
        import litellm  # noqa: PLC0415

        last: Exception | None = None
        extra: dict[str, Any] = {}
        if "qwen" in model.lower():
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        for attempt in range(1, 4):
            try:
                r = litellm.completion(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=512,
                    **extra,
                )
                ch = r.choices[0].message
                return (ch.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last = e
                logger.warning("litellm.completion attempt %s/3 failed: %s", attempt, e)
                if attempt < 3:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
        assert last is not None
        raise last

    async def _complete_async(self, messages: list[dict[str, str]], model: str) -> str:
        """Async LLM call with retries."""
        import litellm  # noqa: PLC0415

        last: Exception | None = None
        for attempt in range(1, 4):
            try:
                r = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                )
                ch = r.choices[0].message
                return (ch.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last = e
                logger.warning("litellm.acompletion attempt %s/3 failed: %s", attempt, e)
                if attempt < 3:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
        assert last is not None
        raise last

    def answer(self, question: str, context: list[dict], *, model: str = "gpt-4o-mini") -> str:
        """Generate a short answer from retrieved ``context`` using LiteLLM."""
        parts = []
        for i, c in enumerate(context, start=1):
            t = (c.get("content") or "").strip()
            if t:
                parts.append(f"[{i}] {t}")
        ctx_joined = "\n\n".join(parts)
        user_prompt = (
            "Based on the following conversation excerpts, answer the question concisely.\n"
            'If the information is not available in the context, say "No information available."\n\n'
            f"Context:\n{ctx_joined}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": user_prompt}]
        return self._complete_sync(messages, model)

    def cleanup(self) -> None:
        """Remove temp data directory and restore ``ANIMAWORKS_DATA_DIR``."""
        if self._temp_dir and Path(self._temp_dir).exists():
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=False)
            except OSError as e:
                logger.warning("Failed to rmtree %s: %s", self._temp_dir, e)
        self._temp_dir = None
        if self._own_data_env:
            if self._previous_animaworks_data is not None:
                os.environ["ANIMAWORKS_DATA_DIR"] = self._previous_animaworks_data
            else:
                os.environ.pop("ANIMAWORKS_DATA_DIR", None)
        self._own_data_env = False
        self._vector_store = None
        self._indexer = None
        self._retriever = None
        self._anima_dir = None
        self._episodes_dir = None

    def __enter__(self) -> AnimaWorksLoCoMoAdapter:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.cleanup()
