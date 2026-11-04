from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Dedicated threads for blocking torch work.

libomp gives every thread that enters a parallel region its own team of
workers, so torch called from many Python threads holds one team per caller.
Running all model work on these threads leaves a single team per lane, however
many threads are waiting for a result.

Each component has its own lane, so embedding and reranking can overlap on CPU
and CUDA as they always have.  Everything on MPS shares one lane: torch's MPS
stream is not safe to use from two threads at once, and doing so aborts the
process with a Metal assertion that Python cannot catch.
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_NUM_THREADS = 4
_DEFAULT_NUM_INTEROP_THREADS = 1

_lanes: dict[str, ThreadPoolExecutor] = {}
_lanes_lock = threading.Lock()
_current = threading.local()
_torch_threads_configured = False
_torch_threads_lock = threading.Lock()


def _configured_thread_limits() -> tuple[int, int]:
    try:
        from core.config import load_config

        gpu = load_config().gpu
        return int(gpu.torch_num_threads), int(gpu.torch_num_interop_threads)
    except Exception:
        return _DEFAULT_NUM_THREADS, _DEFAULT_NUM_INTEROP_THREADS


def configure_torch_threads() -> None:
    """Bound torch's CPU thread pools once, before any of them start.

    ``set_num_threads`` also becomes the starting value for threads that enter
    torch later.  ``set_num_interop_threads`` raises once the inter-op pool has
    started, which only costs the bound, so it is logged and tolerated.
    """
    global _torch_threads_configured

    with _torch_threads_lock:
        if _torch_threads_configured:
            return
        _torch_threads_configured = True
        try:
            import torch
        except ImportError:
            return
        num_threads, num_interop_threads = _configured_thread_limits()
        if num_threads > 0:
            torch.set_num_threads(num_threads)
        if num_interop_threads > 0:
            try:
                torch.set_num_interop_threads(num_interop_threads)
            except RuntimeError as exc:
                logger.warning("torch inter-op threads could not be limited: %s", exc)
        logger.info("torch threads: intra-op=%d inter-op=%d", torch.get_num_threads(), torch.get_num_interop_threads())


def _initialize_lane(lane: str) -> None:
    # An exception here would break the pool and every later inference call.
    _current.lane = lane
    try:
        configure_torch_threads()
    except Exception:
        logger.warning("Failed to configure torch threads", exc_info=True)


def _lane_for(component: str, device: str) -> str:
    return "mps" if device == "mps" else component


def _executor_for(lane: str) -> ThreadPoolExecutor:
    with _lanes_lock:
        executor = _lanes.get(lane)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"inference-{lane}",
                initializer=_initialize_lane,
                initargs=(lane,),
            )
            _lanes[lane] = executor
        return executor


def run_inference(component: str, device: str, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` on the inference thread for ``component`` on ``device`` and wait for it.

    Everything that loads or calls a torch model must come through here, or it
    grows a thread team of its own and, on MPS, can abort the process.
    """
    lane = _lane_for(component, device)
    if getattr(_current, "lane", None) == lane:
        return fn(*args, **kwargs)
    return _executor_for(lane).submit(fn, *args, **kwargs).result()


def reset_inference_for_testing() -> None:
    """Forget lanes and the applied thread limits for test isolation."""
    global _torch_threads_configured

    with _lanes_lock:
        executors = list(_lanes.values())
        _lanes.clear()
    for executor in executors:
        executor.shutdown(wait=False)
    with _torch_threads_lock:
        _torch_threads_configured = False
