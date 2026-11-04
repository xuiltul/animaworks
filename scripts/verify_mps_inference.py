#!/usr/bin/env python3
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Real-model checks for embedding and reranking on Apple MPS.  Not run in CI.

Run on a Mac, one subcommand at a time, from the repository root::

    python scripts/verify_mps_inference.py parity
    python scripts/verify_mps_inference.py concurrency [--raw] [--seconds 40]
    python scripts/verify_mps_inference.py memory [--batch-size 8] [--max-seq 2048]
    python scripts/verify_mps_inference.py fallback

``parity`` compares CPU and MPS output, because vector indexes built on one
device are searched with queries encoded on the other.  ``concurrency`` runs
embedding and reranking together from separate threads and checks every result
against a single-threaded baseline; ``--raw`` calls the models directly from
those threads, without the project's inference threads, which shows what they
protect against: expect the process to abort.  ``memory`` measures the longest batch the configured
limits allow.  ``fallback`` forces a real MPS out-of-memory error and checks that
embedding and reranking carry on from CPU with the same results.

Models are read from their local snapshots, so nothing is downloaded and the
data directory is not written to.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

EMBEDDING_MODEL = "cl-nagoya/ruri-v3-310m"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
SNAPSHOT_ROOTS = (
    Path.home() / ".animaworks" / "models",
    Path.home() / ".cache" / "huggingface" / "hub",
)

HAND_WRITTEN = [
    "昨日の会議で決まった対応事項は何か",
    "おはようございます",
    "はい",
    "。",
    "あいうえおかきくけこさしすせそ",
    "カタカナダケノブンショウハドウナルカ",
    "東京特許許可局局長今日急遽休暇許可拒否",
    "第十二条 甲は乙に対し、本契約に基づく債務の履行を遅滞したときは、年十四・六パーセントの割合による遅延損害金を支払う。",
    "AnimaWorks の RAG は ChromaDB に embedding を保存し、cross-encoder で rerank します。",
    "git push origin main が rejected になったら fetch して rebase してください。",
    "患者さんの血圧は 128/76 mmHg、脈拍 72 回/分で、前回と比べて大きな変化はありません。",
    "2026年11月4日 12時30分 JST、サーバーの負荷平均は 44.95 でした。",
    "https://example.com/path?a=1&b=2 を開いて、エラーの詳細を確認する",
    "🎉✨ やったー！リリース完了！🚀",
    "def add(a: int, b: int) -> int:\n    return a + b  # 足し算",
    "| 列1 | 列2 |\n|---|---|\n| 値A | 値B |",
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
    "今日は天気がいいので、散歩に行こうと思います。夕方から雨が降るそうなので傘を持っていきます。",
    "The quick brown fox jumps over the lazy dog.",
    "한국어 문장도 섞여 있을 수 있습니다。",
    " ",
    "？？？！！！",
    "０１２３４５６７８９ ＡＢＣＤＥＦ",
    "顧客から請求書の再発行依頼があり、経理担当へ転送しました。期限は今週金曜日です。",
    "タスクが blocked のまま 3 日経過している。担当者に確認して、必要なら再割り当てする。",
]

QUERIES = [
    "昨日の会議で決まった対応事項は何か",
    "記憶の検索はどのように動きますか",
    "タスクの委譲と進捗確認の方法",
    "設定ファイルの場所を教えてください",
    "エラーが出たときの対応手順",
    "How do I configure the embedding model?",
    "ハートビートの実行間隔",
    "Slack の通知を止めたい",
    "権限はどこで設定しますか",
    "音声入力の使い方",
]


def corpus_from_docs(limit: int) -> list[str]:
    """Build varied Japanese text from the repository's own documents."""
    files = sorted((PROJECT_DIR / "docs" / "ja").rglob("*.md"))
    rng = random.Random(20261104)
    rng.shuffle(files)
    texts: list[str] = []
    lengths = (40, 120, 300, 900, 2500, 8000)
    for path in files:
        body = path.read_text(encoding="utf-8")
        if len(body) < 200:
            continue
        for length in lengths:
            start = rng.randrange(0, max(1, len(body) - length))
            texts.append(body[start : start + length])
        if len(texts) >= limit:
            break
    return texts[:limit]


def snapshot_path(name: str) -> str:
    folder = "models--" + name.replace("/", "--")
    for root in SNAPSHOT_ROOTS:
        found = sorted((root / folder / "snapshots").glob("*"))
        if found:
            return str(found[-1])
    raise SystemExit(f"No local snapshot of {name} under {SNAPSHOT_ROOTS}")


def load_embedder(device: str, max_seq: int) -> Any:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(snapshot_path(EMBEDDING_MODEL), device=device)
    if isinstance(model.max_seq_length, int) and model.max_seq_length > max_seq:
        model.max_seq_length = max_seq
    return model


def load_reranker(device: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(snapshot_path(RERANK_MODEL), device=device)


def encode(model: Any, texts: list[str], batch_size: int) -> Any:
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)


def score(model: Any, pairs: list[list[str]]) -> Any:
    return model.predict(pairs, show_progress_bar=False)


def use_project_stack() -> Any:
    """Point the project's code at local snapshots, with the live device settings and a temporary data directory."""
    os.environ["ANIMAWORKS_DATA_DIR"] = tempfile.mkdtemp(prefix="animaworks-mps-check-")
    for name in ("ANIMAWORKS_EMBED_URL", "ANIMAWORKS_RERANK_URL", "ANIMAWORKS_VECTOR_URL"):
        os.environ.pop(name, None)

    import core.config
    from core.config.schemas import AnimaWorksConfig, GPUConfig, RAGConfig

    config = AnimaWorksConfig(
        rag=RAGConfig(
            use_gpu=True,
            embedding_model=snapshot_path(EMBEDDING_MODEL),
            cross_encoder_model=snapshot_path(RERANK_MODEL),
        ),
        gpu=GPUConfig(embedding_device="auto", reranker_device="cuda", embedding_batch_size=8),
    )
    core.config.load_config = lambda *_args, **_kwargs: config
    return config


# ── parity ──────────────────────────────────────────────────


def cmd_parity(args: argparse.Namespace) -> int:
    import numpy as np

    texts = HAND_WRITTEN + corpus_from_docs(args.documents)
    print(f"texts={len(texts)} chars min={min(map(len, texts))} max={max(map(len, texts))}", flush=True)
    cpu, mps = load_embedder("cpu", args.max_seq), load_embedder("mps", args.max_seq)

    def normalized(values: Any) -> Any:
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    start = time.perf_counter()
    from_cpu = encode(cpu, texts, args.batch_size)
    cpu_seconds = time.perf_counter() - start
    start = time.perf_counter()
    from_mps = encode(mps, texts, args.batch_size)
    mps_seconds = time.perf_counter() - start
    print(f"encode cpu={cpu_seconds:.1f}s mps={mps_seconds:.1f}s", flush=True)

    # A query is encoded alone while documents are batched with others, so also
    # compare a text encoded alone on MPS with the batched CPU vector.
    singles = encode(mps, HAND_WRITTEN + texts[:: max(1, len(texts) // 40)], 1)
    single_refs = np.concatenate([from_cpu[: len(HAND_WRITTEN)], from_cpu[:: max(1, len(texts) // 40)]])

    worst_ok = True
    for label, left, right in (
        ("batched mps vs batched cpu", from_mps, from_cpu),
        ("single  mps vs batched cpu", singles, single_refs),
    ):
        cosines = np.sum(normalized(left) * normalized(right), axis=1)
        print(
            f"cosine[{label}]: n={len(cosines)} min={cosines.min():.8f} "
            f"p1={np.percentile(cosines, 1):.8f} median={np.median(cosines):.8f} "
            f"max_abs_dim_diff={np.abs(left - right).max():.6f}",
            flush=True,
        )
        worst_ok = worst_ok and bool(cosines.min() > args.min_cosine)
    argmin = int(np.argmin(np.sum(normalized(from_mps) * normalized(from_cpu), axis=1)))
    print(f"lowest-cosine text ({len(texts[argmin])} chars): {texts[argmin][:60]!r}", flush=True)

    # Search compatibility: MPS query vectors against a CPU-built index.
    index = normalized(from_cpu)
    query_cpu = normalized(encode(cpu, QUERIES, 1))
    query_mps = normalized(encode(mps, QUERIES, 1))
    identical = overlap = 0
    for left, right in zip(query_cpu, query_mps, strict=True):
        top_cpu = np.argsort(-(index @ left))[:10].tolist()
        top_mps = np.argsort(-(index @ right))[:10].tolist()
        identical += top_cpu == top_mps
        overlap += len(set(top_cpu) & set(top_mps))
    print(f"search top-10: identical order {identical}/{len(QUERIES)}, overlap {overlap}/{10 * len(QUERIES)}")

    # Rerank ordering.
    del cpu, mps
    ce_cpu, ce_mps = load_reranker("cpu"), load_reranker("mps")
    rng = random.Random(7)
    same_order = same_top5 = 0
    max_diff = 0.0
    taus: list[float] = []
    for query in QUERIES:
        docs = rng.sample(texts, 30)
        pairs = [[query, doc] for doc in docs]
        left, right = np.asarray(score(ce_cpu, pairs)), np.asarray(score(ce_mps, pairs))
        order_left, order_right = np.argsort(-left).tolist(), np.argsort(-right).tolist()
        same_order += order_left == order_right
        same_top5 += order_left[:5] == order_right[:5]
        max_diff = max(max_diff, float(np.abs(left - right).max()))
        taus.append(kendall_tau(order_left, order_right))
    print(
        f"rerank: 30 docs x {len(QUERIES)} queries, identical order {same_order}/{len(QUERIES)}, "
        f"identical top-5 {same_top5}/{len(QUERIES)}, min kendall tau {min(taus):.4f}, "
        f"max abs score diff {max_diff:.6f}",
        flush=True,
    )
    ok = worst_ok and same_top5 == len(QUERIES)
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


def kendall_tau(left: list[int], right: list[int]) -> float:
    rank = {value: pos for pos, value in enumerate(right)}
    ranks = [rank[value] for value in left]
    size = len(ranks)
    concordant = sum(ranks[i] < ranks[j] for i in range(size) for j in range(i + 1, size))
    pairs = size * (size - 1) // 2
    return (2 * concordant - pairs) / pairs


# ── concurrency ─────────────────────────────────────────────


def cmd_concurrency(args: argparse.Namespace) -> int:
    import numpy as np

    texts = HAND_WRITTEN + corpus_from_docs(40)
    embed_batches = [texts[i : i + 8] for i in range(0, len(texts), 8)]
    pair_batches = [[[query, doc] for doc in random.Random(i).sample(texts, 30)] for i, query in enumerate(QUERIES)]

    if args.raw:
        embedder, reranker = load_embedder("mps", 2048), load_reranker("mps")

        def run_embed(batch: list[str]) -> Any:
            return encode(embedder, batch, 8)

        def run_rerank(pairs: list[list[str]]) -> Any:
            return score(reranker, pairs)

        mode = "raw model calls"
    else:
        config = use_project_stack()

        from core.memory.rag.embedding import get_embedding_model, thread_safe_encode
        from core.memory.retrieval.reranker import get_reranker

        reranker = get_reranker(config.rag.cross_encoder_model)

        def run_embed(batch: list[str]) -> Any:
            return np.asarray(thread_safe_encode(batch))

        def run_rerank(pairs: list[list[str]]) -> Any:
            return reranker.score_sync(pairs[0][0], [pair[1] for pair in pairs])

        mode = "thread_safe_encode and score_sync"

    print(f"mode: {mode}", flush=True)
    baseline_embed = [run_embed(batch) for batch in embed_batches]
    baseline_rerank = [run_rerank(pairs) for pairs in pair_batches]
    if not args.raw:
        print(f"devices: embedding={get_embedding_model().device} reranker={reranker._model.device}", flush=True)

    stop_at = time.monotonic() + args.seconds
    problems: list[str] = []
    counts = {"embed": 0, "rerank": 0}
    lock = threading.Lock()

    def worker(kind: str, seed: int) -> None:
        rng = random.Random(seed)
        while time.monotonic() < stop_at and not problems:
            try:
                if kind == "embed":
                    index = rng.randrange(len(embed_batches))
                    got, want = run_embed(embed_batches[index]), baseline_embed[index]
                    got_unit = got / np.linalg.norm(got, axis=1, keepdims=True)
                    want_unit = want / np.linalg.norm(want, axis=1, keepdims=True)
                    worst = float(np.sum(got_unit * want_unit, axis=1).min())
                    if worst < args.min_cosine:
                        problems.append(f"embed cosine {worst:.8f} on batch {index}")
                else:
                    index = rng.randrange(len(pair_batches))
                    got, want = run_rerank(pair_batches[index]), baseline_rerank[index]
                    worst = float(np.abs(np.asarray(got) - np.asarray(want)).max())
                    if worst > 1e-3:
                        problems.append(f"rerank score diff {worst:.6f} on batch {index}")
            except Exception as exc:  # noqa: BLE001 - report every failure kind
                problems.append(f"{kind}: {type(exc).__name__}: {exc}")
            with lock:
                counts[kind] += 1

    threads = [
        threading.Thread(target=worker, args=(kind, seed))
        for seed, kind in enumerate(["embed"] * args.embed_threads + ["rerank"] * args.rerank_threads)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"completed {counts}; problems={len(problems)}", flush=True)
    for problem in problems[:5]:
        print("  ", problem[:400], flush=True)
    print("CONCURRENCY OK" if not problems else "CONCURRENCY FAILED")
    return 0 if not problems else 1


# ── memory ──────────────────────────────────────────────────


def footprint_bytes() -> int:
    """Physical footprint of this process in bytes, GPU memory included."""
    out = subprocess.run(["footprint", "-p", str(os.getpid()), "--noCategories"], capture_output=True, text=True).stdout
    match = re.search(r"Footprint:\s+([\d.]+)\s*(KB|MB|GB)", out) or re.search(r"([\d.]+)\s*(KB|MB|GB)\s+phys", out)
    if not match:
        return 0
    return int(float(match.group(1)) * {"KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}[match.group(2)])


def peak_while(fn: Callable[[], Any], read: Callable[[], int]) -> tuple[Any, int]:
    peak = [read()]
    done = threading.Event()

    def poll() -> None:
        while not done.wait(0.05):
            peak[0] = max(peak[0], read())

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    try:
        return fn(), peak[0]
    finally:
        done.set()
        poller.join()


def cmd_memory(args: argparse.Namespace) -> int:
    import torch

    gib = 1 << 30
    model = load_embedder("mps", args.max_seq)
    body = "".join(HAND_WRITTEN[:18]) + "".join(corpus_from_docs(60))
    tokens = model.tokenizer(body, truncation=False)["input_ids"]
    if len(tokens) < args.max_seq * 2:
        raise SystemExit(f"Corpus too short: {len(tokens)} tokens")
    texts = [body[i * 50 :] for i in range(args.batch_size)]
    seen = [len(model.tokenizer(t, truncation=True, max_length=args.max_seq)["input_ids"]) for t in texts]
    print(f"batch={args.batch_size} tokens each={min(seen)}..{max(seen)} (max_seq={args.max_seq})", flush=True)
    print(
        f"recommended_max={torch.mps.recommended_max_memory() / gib:.1f}GiB "
        f"before: current={torch.mps.current_allocated_memory() / gib:.2f}GiB "
        f"driver={torch.mps.driver_allocated_memory() / gib:.2f}GiB",
        flush=True,
    )
    for round_number in range(args.rounds):
        start = time.perf_counter()
        _, peak = peak_while(lambda: encode(model, texts, args.batch_size), torch.mps.driver_allocated_memory)
        print(
            f"round {round_number}: {time.perf_counter() - start:.2f}s peak_driver={peak / gib:.2f}GiB "
            f"after: current={torch.mps.current_allocated_memory() / gib:.2f}GiB "
            f"driver={torch.mps.driver_allocated_memory() / gib:.2f}GiB footprint={footprint_bytes() / gib:.2f}GiB "
            f"maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / gib:.2f}GiB",
            flush=True,
        )
    torch.mps.empty_cache()
    print(
        f"after empty_cache: current={torch.mps.current_allocated_memory() / gib:.2f}GiB "
        f"driver={torch.mps.driver_allocated_memory() / gib:.2f}GiB footprint={footprint_bytes() / gib:.2f}GiB",
        flush=True,
    )

    # Does repeated short work grow memory when it runs off the main thread?
    short = HAND_WRITTEN[:8]
    readings: list[float] = []

    def churn() -> None:
        for step in range(args.churn):
            encode(model, short, 8)
            if step % max(1, args.churn // 5) == 0:
                readings.append(footprint_bytes() / gib)
        readings.append(footprint_bytes() / gib)

    thread = threading.Thread(target=churn)
    thread.start()
    thread.join()
    print(f"footprint during {args.churn} short batches on a worker thread (GiB): {readings}", flush=True)
    return 0


# ── fallback ────────────────────────────────────────────────


def cmd_fallback(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    config = use_project_stack()

    from core.infra.gpu import get_gpu_status
    from core.memory.rag.embedding import get_embedding_model, thread_safe_encode
    from core.memory.retrieval.reranker import get_reranker

    body = "".join(HAND_WRITTEN[:18]) + "".join(corpus_from_docs(60))
    long_texts = [body[i * 50 :] for i in range(8)]
    documents = [text[:400] for text in long_texts]
    query = QUERIES[0]

    reranker = get_reranker(config.rag.cross_encoder_model)
    want_vectors = np.asarray(thread_safe_encode(long_texts))
    want_scores = np.asarray(reranker.score_sync(query, documents))
    print(f"before: {get_gpu_status()} reranker={reranker._device}", flush=True)
    assert get_gpu_status()["embedding_device"] == "mps" and reranker._device == "mps"

    limit = args.limit_gib * (1 << 30)
    torch.mps.empty_cache()
    torch.mps.set_per_process_memory_fraction(limit / torch.mps.recommended_max_memory())
    print(f"MPS limit set to {args.limit_gib}GiB", flush=True)

    # Both components are in flight when memory runs out, which is the awkward case.
    results: dict[str, Any] = {}

    def embed() -> None:
        results["vectors"] = np.asarray(thread_safe_encode(long_texts))

    def rerank() -> None:
        results["scores"] = np.asarray(reranker.score_sync(query, documents))

    start = time.perf_counter()
    threads = [threading.Thread(target=embed), threading.Thread(target=rerank)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"under the limit: {time.perf_counter() - start:.1f}s {get_gpu_status()}", flush=True)
    print(f"embedding now on {get_embedding_model().device}, reranker on {reranker._device}", flush=True)
    print(
        f"MPS memory left: current={torch.mps.current_allocated_memory() / (1 << 30):.2f}GiB "
        f"driver={torch.mps.driver_allocated_memory() / (1 << 30):.2f}GiB",
        flush=True,
    )

    got = results["vectors"]
    cosines = np.sum(got * want_vectors, axis=1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want_vectors, axis=1))
    print(f"embedding cosine against MPS baseline: min={cosines.min():.8f}", flush=True)
    score_diff = float(np.abs(results["scores"] - want_scores).max())
    print(f"rerank max abs score diff against MPS baseline: {score_diff:.6f}", flush=True)

    ok = get_gpu_status()["degraded"] is True and get_embedding_model().device.type == "cpu"
    ok = ok and bool(cosines.min() > 0.9999) and score_diff < 1e-3
    print("FALLBACK OK" if ok else "FALLBACK FAILED")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    parity = commands.add_parser("parity")
    parity.add_argument("--documents", type=int, default=150)
    parity.add_argument("--batch-size", type=int, default=8)
    parity.add_argument("--max-seq", type=int, default=2048)
    parity.add_argument("--min-cosine", type=float, default=0.9999)
    parity.set_defaults(handler=cmd_parity)

    concurrency = commands.add_parser("concurrency")
    concurrency.add_argument("--raw", action="store_true")
    concurrency.add_argument("--seconds", type=float, default=40.0)
    concurrency.add_argument("--embed-threads", type=int, default=2)
    concurrency.add_argument("--rerank-threads", type=int, default=2)
    concurrency.add_argument("--min-cosine", type=float, default=0.9999)
    concurrency.set_defaults(handler=cmd_concurrency)

    memory = commands.add_parser("memory")
    memory.add_argument("--batch-size", type=int, default=8)
    memory.add_argument("--max-seq", type=int, default=2048)
    memory.add_argument("--rounds", type=int, default=3)
    memory.add_argument("--churn", type=int, default=300)
    memory.set_defaults(handler=cmd_memory)

    fallback = commands.add_parser("fallback")
    fallback.add_argument("--limit-gib", type=float, default=1.3)
    fallback.set_defaults(handler=cmd_fallback)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
