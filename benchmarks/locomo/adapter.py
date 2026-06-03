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

from benchmarks.locomo.answer_prompt import (
    ANSWER_SYSTEM as _ANSWER_SYSTEM,
)
from benchmarks.locomo.answer_prompt import (
    LOCOMO_ANSWER_MAX_TOKENS,
    build_answer_user_content,
    merge_pipeline_gate_settings,
    normalize_locomo_answer,
)

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

# ── LLM helpers ──────────

from benchmarks.locomo.llm_config import default_answer_model, resolve_locomo_litellm_kwargs

# ── load_dataset ──────────

_EVENT_METADATA_FIELDS: tuple[str, ...] = (
    "fact_id",
    "valid_at",
    "event_time_iso",
    "event_time_text",
    "session_index",
    "turn_index",
    "sentence_index",
    "speaker",
    "source_episode",
    "event_time_parse_error",
    "entities",
    "confidence",
    "base_score",
    "temporal_boost",
    "entity_boost",
    "entity_overlap",
    "query_entities",
    "candidate_entities",
)


def locomo_temporal_boost_enabled() -> bool:
    """Return True when LoCoMo temporal boost ablation is explicitly enabled."""
    return os.environ.get("LOCOMO_TEMPORAL_BOOST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def locomo_entity_boost_enabled() -> bool:
    """Return True when LoCoMo entity boost ablation is explicitly enabled."""
    return os.environ.get("LOCOMO_ENTITY_BOOST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def locomo_fact_index_enabled() -> bool:
    """Return True when LoCoMo fact dual-index ablation is explicitly enabled."""
    return os.environ.get("LOCOMO_FACT_INDEX", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def locomo_entity_aware_graph_enabled() -> bool:
    """Return True when LoCoMo entity-aware graph ablation is explicitly enabled."""
    return os.environ.get("LOCOMO_ENTITY_AWARE_GRAPH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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


def _conversation_speaker_names(conversation: dict[str, Any]) -> tuple[str, ...]:
    """Return conversation speaker names to ignore in entity boost scoring."""
    names: set[str] = set()
    for key in ("speaker_a", "speaker_b"):
        value = str(conversation.get(key, "") or "").strip()
        if value:
            names.add(value)
    return tuple(sorted(names))


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
        self._facts_dir: Path | None = None
        self._vector_store: Any = None
        self._temp_worker: Any = None
        self._indexer: Any = None
        self._retriever: Any = None
        self._bm25_corpus: list[tuple[str, dict[str, Any]]] | None = None
        self._bm25_index: Any = None
        self._fact_bm25_corpus: list[tuple[str, dict[str, Any]]] | None = None
        self._fact_bm25_index: Any = None
        self._fact_metadata_by_source_file: dict[str, dict[str, Any]] = {}
        self._last_fact_count: int = 0
        self._last_abstain: bool = False
        self._last_abstain_reason: str = ""
        self._last_top_score: float | None = None
        self._last_top_event_time_iso: str = ""
        self._last_top_memory_type: str = ""
        self._last_raw_answer: str = ""
        self._last_normalized_answer: str = ""
        self._entity_ignored_entities: tuple[str, ...] = ()
        # Deferred heavy init
        self._init_isolated_rag()

    def _init_isolated_rag(self) -> None:
        """Create temp data dir, env override, and RAG components."""
        from core.memory.rag.indexer import MemoryIndexer  # noqa: PLC0415
        from core.memory.rag.retriever import MemoryRetriever  # noqa: PLC0415
        from core.memory.rag.singleton import get_embedding_model, get_vector_store  # noqa: PLC0415
        from core.memory.rag.vector_worker_client import start_temporary_vector_worker  # noqa: PLC0415

        self._previous_animaworks_data = os.environ.get("ANIMAWORKS_DATA_DIR")
        real_data_dir = Path(os.environ.get("ANIMAWORKS_DATA_DIR", "~/.animaworks")).expanduser().resolve()
        self._temp_dir = tempfile.mkdtemp(prefix="animaworks-locomo-")
        os.environ["ANIMAWORKS_DATA_DIR"] = self._temp_dir
        self._own_data_env = True
        self._anima_dir = Path(self._temp_dir) / "animas" / ANIMA_NAME
        self._episodes_dir = self._anima_dir / "episodes"
        self._facts_dir = self._anima_dir / "facts"
        for sub in (
            "episodes",
            "facts",
            "knowledge",
            "procedures",
            "common_knowledge",
        ):
            (self._anima_dir / sub).mkdir(parents=True, exist_ok=True)
        real_models = real_data_dir / "models"
        tmp_models = Path(self._temp_dir) / "models"
        if real_models.is_dir() and not tmp_models.exists():
            tmp_models.symlink_to(real_models)

        (self._anima_dir / "vectordb").mkdir(parents=True, exist_ok=True)
        try:
            self._temp_worker = start_temporary_vector_worker(log_dir=Path(self._temp_dir) / "logs")
            self._vector_store = get_vector_store(ANIMA_NAME)
            if self._vector_store is None:
                raise RuntimeError("vector worker unavailable")
        except Exception as e:
            logger.error("Vector worker store init failed: %s", e)
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
        assert (
            self._vector_store is not None
            and self._anima_dir is not None
            and self._episodes_dir is not None
            and self._facts_dir is not None
        )
        for name in self._vector_store.list_collections():
            if name.startswith(ANIMA_NAME):
                self._vector_store.delete_collection(name)
        if self._episodes_dir.exists():
            shutil.rmtree(self._episodes_dir)
        self._episodes_dir.mkdir(parents=True, exist_ok=True)
        if self._facts_dir.exists():
            shutil.rmtree(self._facts_dir)
        self._facts_dir.mkdir(parents=True, exist_ok=True)
        self._bm25_corpus = None
        self._bm25_index = None
        self._fact_bm25_corpus = None
        self._fact_bm25_index = None
        self._fact_metadata_by_source_file = {}
        self._last_fact_count = 0
        if self._retriever is not None:
            self._retriever._knowledge_graph = None
            self._retriever._knowledge_graph_signature = None
        graph_cache = self._anima_dir / "vectordb" / "knowledge_graph.json"
        if graph_cache.exists():
            try:
                graph_cache.unlink()
            except OSError as e:
                logger.warning("Failed to remove %s: %s", graph_cache, e)
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
        assert self._indexer is not None and self._episodes_dir is not None and self._facts_dir is not None
        sample_id = sample.get("sample_id", "unknown")
        conv = sample.get("conversation")
        if not isinstance(conv, dict):
            raise TypeError("sample['conversation'] must be a dict")
        self._entity_ignored_entities = _conversation_speaker_names(conv)
        stem = _episode_stem_for_sample(str(sample_id))
        md = _build_episode_markdown(str(sample_id), conv)
        file_path = self._episodes_dir / f"{stem}.md"
        file_path.write_text(md, encoding="utf-8")
        n = self._indexer.index_file(file_path, memory_type="episodes", force=True)
        self._bm25_corpus = None
        self._bm25_index = None
        self._fact_bm25_corpus = None
        self._fact_bm25_index = None
        self._fact_metadata_by_source_file = {}
        self._last_fact_count = 0
        if locomo_fact_index_enabled():
            self._ingest_fact_index(str(sample_id), conv, source_episode=f"episodes/{stem}.md")
        return n

    def _clear_fact_index_storage(self) -> None:
        """Clear optional fact files, vectors, and in-memory caches before rebuild."""
        assert self._facts_dir is not None
        if self._facts_dir.exists():
            shutil.rmtree(self._facts_dir)
        self._facts_dir.mkdir(parents=True, exist_ok=True)
        vector_store = getattr(self, "_vector_store", None)
        if vector_store is not None:
            try:
                vector_store.delete_collection(f"{ANIMA_NAME}_facts")
            except Exception:
                logger.debug("No LoCoMo facts collection to clear", exc_info=True)
        self._fact_bm25_corpus = []
        self._fact_bm25_index = None
        self._fact_metadata_by_source_file = {}
        self._last_fact_count = 0

    def _ingest_fact_index(self, sample_id: str, conversation: dict[str, Any], *, source_episode: str) -> None:
        """Build and index optional LoCoMo fact memories without failing episode ingest."""
        assert self._indexer is not None and self._facts_dir is not None
        try:
            self._clear_fact_index_storage()
            from benchmarks.locomo.fact_index import (  # noqa: PLC0415
                extract_locomo_fact_records,
                fact_bm25_documents,
                write_fact_records,
            )

            records = extract_locomo_fact_records(sample_id, conversation, source_episode=source_episode)
            if not records:
                self._last_fact_count = 0
                self._fact_bm25_corpus = []
                self._fact_metadata_by_source_file = {}
                return

            write_fact_records(self._facts_dir, records)
            indexed = 0
            for fact_file in sorted(self._facts_dir.glob("fact_*.md")):
                indexed += self._indexer.index_file(fact_file, memory_type="facts", force=True)
            self._last_fact_count = indexed
            self._fact_bm25_corpus = fact_bm25_documents(records)
            self._fact_bm25_index = None
            self._fact_metadata_by_source_file = {
                str(meta.get("source_file", "")): dict(meta)
                for _, meta in self._fact_bm25_corpus
                if meta.get("source_file")
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("LoCoMo fact index skipped after failure: %s", e)
            self._last_fact_count = 0
            self._fact_bm25_corpus = []
            self._fact_bm25_index = None
            self._fact_metadata_by_source_file = {}

    def _retrieval_to_dicts(self, results: list[Any]) -> list[dict[str, Any]]:
        from core.memory.rag.retriever import RetrievalResult  # noqa: PLC0415

        out: list[dict[str, Any]] = []
        for r in results:
            if not isinstance(r, RetrievalResult):
                continue
            meta = r.metadata if isinstance(r.metadata, dict) else {}
            enriched_meta = self._enrich_fact_metadata(dict(meta))
            out.append(
                {
                    "content": r.content,
                    "score": float(r.score),
                    "metadata": enriched_meta,
                },
            )
        return out

    def _enrich_fact_metadata(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Attach adapter-side fact metadata omitted by the generic indexer."""
        source_file = str(meta.get("source_file", "") or "")
        if not source_file:
            return meta
        fact_meta = self._fact_metadata_by_source_file.get(source_file)
        if not fact_meta:
            return meta
        return {**fact_meta, **meta, "memory_type": "facts"}

    def _pipeline_item_from_adapter_hit(self, item: dict[str, Any]) -> dict[str, Any]:
        """Normalize adapter retrieval dict for ``RetrievalPipeline``."""
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        out: dict[str, Any] = {
            "content": (item.get("content") or "").strip(),
            "score": float(item.get("score", 0.0)),
            "source_file": str(meta.get("source_file", meta.get("source", ""))),
            "chunk_index": int(meta.get("section", meta.get("chunk_index", 0))),
            "memory_type": str(meta.get("memory_type", item.get("memory_type", "episodes")) or "episodes"),
            "search_method": str(meta.get("search_method", "")),
        }
        for key in _EVENT_METADATA_FIELDS:
            if key in meta:
                out[key] = meta[key]
        return out

    def _adapter_hit_from_pipeline_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Convert pipeline output back to LoCoMo adapter context shape."""
        metadata: dict[str, Any] = {
            "source_file": item.get("source_file", ""),
            "chunk_index": item.get("chunk_index", 0),
            "memory_type": item.get("memory_type", "episodes"),
            "search_method": item.get("search_method", "pipeline"),
        }
        for key in _EVENT_METADATA_FIELDS:
            if key in item:
                metadata[key] = item[key]
        return {
            "content": item.get("content", ""),
            "score": float(item.get("score", 0.0)),
            "metadata": metadata,
        }

    def _remember_retrieval_diagnostics(self, items: list[dict[str, Any]]) -> None:
        """Store lightweight retrieval diagnostics for benchmark output."""
        if not items:
            self._last_top_score = None
            self._last_top_event_time_iso = ""
            self._last_top_memory_type = ""
            return
        top = max(items, key=lambda item: float(item.get("score", 0.0) or 0.0))
        self._last_top_score = float(top.get("score", 0.0) or 0.0)
        meta = top.get("metadata") if isinstance(top.get("metadata"), dict) else {}
        self._last_top_event_time_iso = str(meta.get("event_time_iso", "") or top.get("event_time_iso", "") or "")
        self._last_top_memory_type = str(meta.get("memory_type", "") or top.get("memory_type", "") or "")

    def _load_pipeline_settings(self) -> dict[str, object]:
        """Resolve RAG pipeline knobs (same defaults as ``RAGMemorySearch``)."""
        defaults: dict[str, object] = {
            "rerank_enabled": True,
            "rerank_candidate_pool": 50,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "abstain_on_low_confidence": True,
            "confidence_threshold": 0.35,
            "rrf_confidence_threshold": 0.02,
        }
        try:
            cfg_path = Path("~/.animaworks/config.json").expanduser()
            if cfg_path.is_file():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                rag = cfg.get("rag", {})
                defaults.update(
                    {
                        "rerank_enabled": rag.get("rerank_enabled", defaults["rerank_enabled"]),
                        "rerank_candidate_pool": rag.get(
                            "rerank_candidate_pool",
                            defaults["rerank_candidate_pool"],
                        ),
                        "cross_encoder_model": rag.get(
                            "cross_encoder_model",
                            defaults["cross_encoder_model"],
                        ),
                        "abstain_on_low_confidence": rag.get(
                            "abstain_on_low_confidence",
                            defaults["abstain_on_low_confidence"],
                        ),
                        "confidence_threshold": rag.get(
                            "confidence_threshold",
                            defaults["confidence_threshold"],
                        ),
                        "rrf_confidence_threshold": rag.get(
                            "rrf_confidence_threshold",
                            defaults["rrf_confidence_threshold"],
                        ),
                    },
                )
        except Exception:
            logger.debug("Using default pipeline settings for LoCoMo adapter", exc_info=True)
        return defaults

    def retrieve(self, question: str, *, category: int | None = None) -> list[dict[str, Any]]:
        """Run retrieval for ``question`` following ``self._search_mode``."""
        self._last_abstain = False
        self._last_abstain_reason = ""
        self._last_top_score = None
        self._last_top_event_time_iso = ""
        self._last_top_memory_type = ""
        assert self._retriever is not None
        if self._search_mode == "vector":
            res = self._retriever.search(
                query=question,
                anima_name=ANIMA_NAME,
                memory_type="episodes",
                top_k=self._top_k,
                enable_spreading_activation=False,
            )
            items = self._retrieval_to_dicts(res)
            self._remember_retrieval_diagnostics(items)
            return items
        if self._search_mode == "vector_graph":
            res = self._retriever.search(
                query=question,
                anima_name=ANIMA_NAME,
                memory_type="episodes",
                top_k=self._top_k,
                enable_spreading_activation=True,
            )
            items = self._retrieval_to_dicts(res)
            self._remember_retrieval_diagnostics(items)
            return items
        # scope_all: production-compatible Legacy unified search with benchmark ablations.
        from core.memory.retrieval.entity import EntityBoostConfig  # noqa: PLC0415
        from core.memory.retrieval.temporal import TemporalBoostConfig  # noqa: PLC0415
        from core.memory.retrieval.unified_search import UnifiedMemorySearch  # noqa: PLC0415

        assert self._anima_dir is not None
        settings = self._load_pipeline_settings()
        gate = merge_pipeline_gate_settings(settings, category=category)
        search_settings = {
            **settings,
            "confidence_threshold": gate["confidence_threshold"],
            "rrf_confidence_threshold": gate["rrf_confidence_threshold"],
        }
        scope_override = (
            ("episodes", "facts") if locomo_fact_index_enabled() and self._last_fact_count > 0 else ("episodes",)
        )
        searcher = UnifiedMemorySearch(
            self._anima_dir,
            common_knowledge_dir=self._anima_dir / "common_knowledge",
            common_skills_dir=self._anima_dir / "common_skills",
        )
        result_items = searcher.search(
            question,
            scope="all",
            limit=self._top_k,
            trigger="chat",
            scope_override=scope_override,
            pipeline_settings=search_settings,
            temporal_boost=TemporalBoostConfig(
                enabled=locomo_temporal_boost_enabled(),
                category=category,
            ),
            entity_boost=EntityBoostConfig(
                enabled=locomo_entity_boost_enabled(),
                category=category,
                ignored_entities=self._entity_ignored_entities,
            ),
        )
        meta = searcher.last_search_meta
        self._last_abstain = bool(meta.get("abstain", False))
        self._last_abstain_reason = str(meta.get("abstain_reason", "") or "")
        items = [self._adapter_hit_from_pipeline_item(x) for x in result_items]
        self._remember_retrieval_diagnostics(items)
        return items

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
                metadata: dict[str, Any] = {"source_file": p.name, "section": j, "memory_type": "episodes"}
                first_line = seg.splitlines()[0].strip() if seg.splitlines() else ""
                if first_line.startswith("## Session"):
                    from core.memory.rag.episode_time import apply_episode_heading_event_time  # noqa: PLC0415

                    apply_episode_heading_event_time(metadata, first_line)
                documents.append((seg, metadata))
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

    def _search_fact_vectors(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Dense vector search over optional fact memory collection."""
        assert self._retriever is not None
        try:
            res = self._retriever.search(
                query=query,
                anima_name=ANIMA_NAME,
                memory_type="facts",
                top_k=top_k,
                enable_spreading_activation=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("LoCoMo fact vector retrieval skipped after failure: %s", e)
            return []
        out = self._retrieval_to_dicts(res)
        for row in out:
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            meta["memory_type"] = "facts"
            meta.setdefault("search_method", "fact_vector")
        return out

    def _build_fact_bm25_cache(self) -> None:
        """Build BM25 index from optional fact records cached at ingest."""
        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        documents = self._fact_bm25_corpus or []
        if documents:
            tokenized = [_bm25_tokenize(doc) for doc, _ in documents]
            self._fact_bm25_index = BM25Okapi(tokenized) if any(tokenized) else None
        else:
            self._fact_bm25_index = None

    def _fact_bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """BM25 keyword search using optional fact corpus."""
        if self._fact_bm25_corpus is None:
            self._fact_bm25_corpus = []
        if self._fact_bm25_index is None:
            self._build_fact_bm25_cache()
        documents = self._fact_bm25_corpus
        if not documents:
            return []
        qtok = _bm25_tokenize(query)
        if not qtok or self._fact_bm25_index is None:
            return [
                {
                    "content": documents[0][0],
                    "score": 0.0,
                    "metadata": {**documents[0][1], "search_method": "fact_bm25_degenerate"},
                }
            ][:1]
        scores = self._fact_bm25_index.get_scores(qtok)
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[: top_k * 2]
        out: list[dict[str, Any]] = []
        for i in order:
            out.append(
                {
                    "content": documents[i][0],
                    "score": float(scores[i]),
                    "metadata": {**documents[i][1], "memory_type": "facts", "search_method": "fact_bm25"},
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

        litellm_model, extra = resolve_locomo_litellm_kwargs(model)
        last: Exception | None = None
        for attempt in range(1, 4):
            try:
                r = litellm.completion(
                    model=litellm_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=LOCOMO_ANSWER_MAX_TOKENS,
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

        litellm_model, extra = resolve_locomo_litellm_kwargs(model)
        last: Exception | None = None
        for attempt in range(1, 4):
            try:
                r = await litellm.acompletion(
                    model=litellm_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=LOCOMO_ANSWER_MAX_TOKENS,
                    **extra,
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

    def answer(
        self,
        question: str,
        context: list[dict],
        *,
        model: str | None = None,
        category: int | None = None,
    ) -> str:
        """Generate a short answer from retrieved ``context`` using LiteLLM."""
        if getattr(self, "_last_abstain", False):
            self._last_raw_answer = ""
            self._last_normalized_answer = "No information available."
            return self._last_normalized_answer
        parts = []
        for i, c in enumerate(context, start=1):
            t = (c.get("content") or "").strip()
            if t:
                parts.append(f"[{i}] {t}")
        ctx_joined = "\n\n".join(parts)
        user_content = build_answer_user_content(
            question,
            ctx_joined,
            category=category,
        )
        messages = [
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        raw = self._complete_sync(messages, model or default_answer_model())
        self._last_raw_answer = raw
        self._last_normalized_answer = normalize_locomo_answer(raw, category=category)
        return self._last_normalized_answer

    def cleanup(self) -> None:
        """Remove temp data directory and restore ``ANIMAWORKS_DATA_DIR``."""
        temp_worker = getattr(self, "_temp_worker", None)
        if temp_worker is not None:
            try:
                temp_worker.stop()
            except Exception as e:
                logger.warning("Failed to stop temporary vector worker: %s", e)
            self._temp_worker = None
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
