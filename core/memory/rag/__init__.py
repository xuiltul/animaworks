from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""RAG (Retrieval-Augmented Generation) subsystem.

Provides dense vector search capabilities with:
- Vector similarity search (semantic)
- Temporal decay scoring
- Knowledge graph spreading activation

Based on: docs/design/implemented/priming-layer-design.md Phase 2-3
"""

from core.memory.rag.graph import KnowledgeGraph, create_knowledge_graph
from core.memory.rag.indexer import MemoryIndexer
from core.memory.rag.retriever import MemoryRetriever
from core.memory.rag.store import ChromaVectorStore, VectorStore
from core.memory.rag.watcher import FileWatcher, create_file_watcher

__all__ = [
    "VectorStore",
    "ChromaVectorStore",
    "MemoryIndexer",
    "MemoryRetriever",
    "KnowledgeGraph",
    "create_knowledge_graph",
    "FileWatcher",
    "create_file_watcher",
]
