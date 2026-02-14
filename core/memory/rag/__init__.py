from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RAG (Retrieval-Augmented Generation) subsystem.

Provides hybrid search capabilities combining:
- Vector similarity search (semantic)
- BM25 keyword search
- Temporal decay scoring
- Knowledge graph spreading activation (Phase 3)

Based on: docs/design/implemented/priming-layer-design.md Phase 2-3
"""

from core.memory.rag.graph import KnowledgeGraph, create_knowledge_graph
from core.memory.rag.indexer import MemoryIndexer
from core.memory.rag.retriever import HybridRetriever
from core.memory.rag.store import ChromaVectorStore, VectorStore
from core.memory.rag.watcher import FileWatcher, create_file_watcher

__all__ = [
    "VectorStore",
    "ChromaVectorStore",
    "MemoryIndexer",
    "HybridRetriever",
    "KnowledgeGraph",
    "create_knowledge_graph",
    "FileWatcher",
    "create_file_watcher",
]
