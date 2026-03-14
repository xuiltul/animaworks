#!/usr/bin/env python3
"""E2E verification of [IMPORTANT] tag across RAG indexer → retriever → forgetting → priming."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.paths import get_animas_dir, get_anima_vectordb_dir
from core.memory.rag.store import ChromaVectorStore
from core.memory.rag.indexer import MemoryIndexer
from core.memory.rag.retriever import MemoryRetriever, WEIGHT_IMPORTANCE
from core.memory.forgetting import ForgettingEngine
from core.memory.priming import PrimingEngine

ANIMA_NAME = "kotoha"
DIVIDER = "=" * 70


def step_header(n: int, title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  STEP {n}: {title}")
    print(DIVIDER)


def main() -> None:
    anima_dir = get_animas_dir() / ANIMA_NAME
    shared_dir = anima_dir.parent.parent / "shared"
    vdb_dir = get_anima_vectordb_dir(ANIMA_NAME)
    knowledge_dir = anima_dir / "knowledge"

    # ── Step 1: Re-index ──
    step_header(1, "RAG Re-indexing (force=True)")
    store = ChromaVectorStore(persist_dir=vdb_dir)
    indexer = MemoryIndexer(store, ANIMA_NAME, anima_dir)
    total = indexer.index_directory(knowledge_dir, "knowledge", force=True)
    print(f"  Indexed {total} chunks from knowledge/")

    # ── Step 2: Verify metadata ──
    step_header(2, "Verify importance metadata on indexed chunks")
    coll_name = f"{ANIMA_NAME}_knowledge"
    coll = store.client.get_or_create_collection(name=coll_name)
    data = coll.get(include=["metadatas", "documents"])

    important_chunks: list[tuple[str, str]] = []
    normal_chunks: list[str] = []

    for i, doc_id in enumerate(data["ids"]):
        meta = data["metadatas"][i] if data["metadatas"] else {}
        content = (data["documents"][i] or "")[:80]
        imp = meta.get("importance", "?")
        if imp == "important":
            important_chunks.append((doc_id, content))
        else:
            normal_chunks.append(doc_id)

    print(f"\n  Total chunks: {len(data['ids'])}")
    print(f"  importance='important': {len(important_chunks)}")
    print(f"  importance='normal':    {len(normal_chunks)}")

    if important_chunks:
        print("\n  [IMPORTANT] chunks found:")
        for doc_id, preview in important_chunks:
            print(f"    ✅ {doc_id}")
            print(f"       {preview}...")
    else:
        print("\n  ❌ NO [IMPORTANT] chunks found! Something is wrong.")
        return

    # ── Step 3: Retriever score boost ──
    step_header(3, "Retriever importance boost (+0.20)")
    retriever = MemoryRetriever(store, indexer, knowledge_dir)

    query = "SQL漏洩 インシデント エスカレーション 失敗"
    print(f"  Query: '{query}'")
    print(f"  Expected WEIGHT_IMPORTANCE = {WEIGHT_IMPORTANCE}")

    results = retriever.search(
        query, ANIMA_NAME, memory_type="knowledge", top_k=10,
    )

    print(f"\n  Top {len(results)} results:")
    boosted_found = False
    for r in results:
        imp_score = r.source_scores.get("importance", 0.0)
        marker = "🔴 BOOSTED" if imp_score > 0 else ""
        print(f"    score={r.score:.4f}  imp_boost={imp_score:.2f}  "
              f"importance={r.metadata.get('importance', '?'):>9s}  "
              f"{marker}")
        print(f"      id: {r.doc_id}")
        print(f"      preview: {r.content[:100].replace(chr(10), ' ')}...")
        if imp_score > 0:
            boosted_found = True

    if boosted_found:
        print("\n  ✅ Importance boost confirmed in search results")
    else:
        print("\n  ❌ No importance-boosted results found")

    # ── Step 4: Forgetting protection ──
    step_header(4, "Forgetting engine protection check")
    engine = ForgettingEngine(anima_dir, ANIMA_NAME)

    protected_count = 0
    unprotected_count = 0
    for i, doc_id in enumerate(data["ids"]):
        meta = data["metadatas"][i] if data["metadatas"] else {}
        is_protected = engine._is_protected(meta)
        imp = meta.get("importance", "?")
        if imp == "important":
            status = "✅ PROTECTED" if is_protected else "❌ NOT PROTECTED"
            print(f"    {doc_id}: importance={imp} → {status}")
            if is_protected:
                protected_count += 1
            else:
                unprotected_count += 1

    if protected_count > 0 and unprotected_count == 0:
        print(f"\n  ✅ All {protected_count} [IMPORTANT] chunks are protected from forgetting")
    else:
        print(f"\n  ❌ {unprotected_count} important chunks are NOT protected!")

    # ── Step 5: Priming injection ──
    step_header(5, "Priming injection test")
    print("  Simulating a message that should trigger recall of the SQL incident...")

    async def test_priming() -> None:
        priming = PrimingEngine(anima_dir, shared_dir)
        message = "外部へのデータ共有について確認したいのですが、過去にインシデントがありましたか？"
        print(f"  Message: '{message}'")

        result = await priming.prime_memories(
            message=message,
            sender_name="human",
            channel="chat",
        )

        channels = {
            "related_knowledge": result.related_knowledge,
            "related_knowledge_untrusted": result.related_knowledge_untrusted,
            "episodes": result.episodes,
            "recent_activity": result.recent_activity,
            "pending_tasks": result.pending_tasks,
            "sender_profile": result.sender_profile,
        }
        print("\n  Priming channels:")
        for name, content in channels.items():
            length = len(content) if content else 0
            has_important = "[IMPORTANT]" in content if content else False
            marker = " ← contains [IMPORTANT]!" if has_important else ""
            print(f"    {name}: {length} chars{marker}")

        print("\n  --- related_knowledge content ---")
        if result.related_knowledge:
            print(f"  {result.related_knowledge[:1000]}")
        else:
            print("  (empty)")

        full_text = (result.related_knowledge or "") + (result.episodes or "")
        if "[IMPORTANT]" in full_text or "インシデント" in full_text or "SQL" in full_text:
            print(f"\n  ✅ Priming successfully recalled [IMPORTANT]-tagged knowledge!")
        else:
            print(f"\n  ❌ [IMPORTANT] knowledge was NOT found in priming output")

    asyncio.run(test_priming())

    # ── Summary ──
    print(f"\n{DIVIDER}")
    print("  E2E SUMMARY")
    print(DIVIDER)
    print(f"  Step 1 (Indexing):    {total} chunks indexed")
    print(f"  Step 2 (Metadata):    {len(important_chunks)} chunks with importance=important")
    print(f"  Step 3 (Boost):       {'✅' if boosted_found else '❌'} +{WEIGHT_IMPORTANCE} score boost")
    print(f"  Step 4 (Forgetting):  {'✅' if protected_count > 0 else '❌'} {protected_count} protected")
    print(f"  Step 5 (Priming):     (see above)")
    print()


if __name__ == "__main__":
    main()
