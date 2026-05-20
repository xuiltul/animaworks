"""Index per-anima memory collections one by one with explicit persist."""
from __future__ import annotations

import gc
import logging
import sys

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

sys.path.insert(0, "/home/main/dev/animaworks")

from core.memory.rag import MemoryIndexer  # noqa: E402
from core.memory.rag.singleton import get_vector_store  # noqa: E402
from core.memory.rag.vector_worker_client import start_temporary_vector_worker  # noqa: E402
from core.paths import get_anima_vectordb_dir, get_data_dir  # noqa: E402

MEMORY_TYPES = ["knowledge", "episodes", "procedures", "skills"]
EXPECTED = {"knowledge", "episodes", "procedures", "skills", "conversation_summaries"}

base = get_data_dir()
animas_dir = base / "animas"

names = sorted(d.name for d in animas_dir.iterdir() if d.is_dir())
grand_total = 0
worker = start_temporary_vector_worker()
try:
    for name in names:
        ad = animas_dir / name
        vdb_dir = get_anima_vectordb_dir(name)
        if not vdb_dir.exists():
            print(f"{name}: NO VECTORDB, skip", flush=True)
            continue

        vs = get_vector_store(name)
        if vs is None:
            print(f"{name}: VECTOR WORKER UNAVAILABLE, skip", flush=True)
            continue
        existing = set(vs.list_collections())
        need = [mt for mt in MEMORY_TYPES if mt not in existing and (ad / mt).is_dir()]
        conv_file = ad / "state" / "conversation.json"
        need_conv = "conversation_summaries" not in existing and conv_file.is_file()

        if not need and not need_conv:
            print(f"{name}: OK (already indexed)", flush=True)
            continue

        ix = MemoryIndexer(vs, name, ad)
        total = 0
        for mt in need:
            c = ix.index_directory(ad / mt, mt, force=True)
            total += c
            print(f"{name}: {mt} -> {c} chunks", flush=True)

        if need_conv:
            c = ix.index_conversation_summary(ad / "state", name, force=True)
            total += c
            print(f"{name}: conversation_summaries -> {c} chunks", flush=True)

        print(f"{name}: DONE ({total} total)", flush=True)
        grand_total += total

        del ix, vs
        gc.collect()
finally:
    worker.stop()

print(f"=== ALL DONE === ({grand_total} total chunks)", flush=True)
