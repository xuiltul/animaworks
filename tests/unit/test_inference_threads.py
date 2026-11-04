"""Tests for the dedicated torch threads and the bounded chroma pools."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config.schemas import AnimaWorksConfig, GPUConfig, RAGConfig
from core.infra.inference import configure_torch_threads, reset_inference_for_testing, run_inference

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fresh_lanes(monkeypatch: pytest.MonkeyPatch):
    """Isolate lane state and keep the real torch out of these tests."""
    reset_inference_for_testing()
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    yield
    reset_inference_for_testing()


def _fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.set_num_threads = MagicMock()  # type: ignore[attr-defined]
    torch.set_num_interop_threads = MagicMock()  # type: ignore[attr-defined]
    torch.get_num_threads = MagicMock(return_value=4)  # type: ignore[attr-defined]
    torch.get_num_interop_threads = MagicMock(return_value=1)  # type: ignore[attr-defined]
    return torch


def _current_thread_id() -> int:
    return threading.get_ident()


class TestRunInference:
    def test_returns_result_and_passes_arguments(self) -> None:
        assert run_inference("embedding", "cpu", lambda a, b=0: a + b, 2, b=3) == 5

    def test_propagates_exceptions(self) -> None:
        def fail() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            run_inference("embedding", "cpu", fail)

    def test_many_callers_share_one_thread(self) -> None:
        """Torch must see one thread however many threads are waiting on it."""
        with ThreadPoolExecutor(max_workers=16) as callers:
            ids = set(callers.map(lambda _: run_inference("embedding", "cpu", _current_thread_id), range(64)))
        assert len(ids) == 1
        assert ids != {threading.get_ident()}

    def test_components_overlap_on_cpu_and_cuda(self) -> None:
        for device in ("cpu", "cuda"):
            embedding = run_inference("embedding", device, _current_thread_id)
            reranker = run_inference("reranker", device, _current_thread_id)
            assert embedding != reranker

    def test_all_mps_work_shares_one_thread(self) -> None:
        assert run_inference("embedding", "mps", _current_thread_id) == run_inference(
            "reranker", "mps", _current_thread_id
        )

    def test_mps_and_cpu_do_not_share_a_thread(self) -> None:
        assert run_inference("embedding", "mps", _current_thread_id) != run_inference(
            "reranker", "cpu", _current_thread_id
        )

    def test_mps_work_never_overlaps(self) -> None:
        running = 0
        most = 0
        guard = threading.Lock()

        def work(_: int) -> None:
            nonlocal running, most
            with guard:
                running += 1
                most = max(most, running)
            time.sleep(0.002)
            with guard:
                running -= 1

        def call(index: int) -> None:
            run_inference("embedding" if index % 2 else "reranker", "mps", work, index)

        with ThreadPoolExecutor(max_workers=8) as callers:
            list(callers.map(call, range(40)))
        assert most == 1

    def test_nested_call_on_the_same_lane_does_not_deadlock(self) -> None:
        def outer() -> int:
            return run_inference("reranker", "mps", _current_thread_id)

        assert run_inference("embedding", "mps", outer) == run_inference("embedding", "mps", _current_thread_id)


class TestConfigureTorchThreads:
    def _configure(self, **gpu_settings: int) -> types.ModuleType:
        config = AnimaWorksConfig(gpu=GPUConfig(**gpu_settings))
        with patch("core.config.load_config", return_value=config):
            configure_torch_threads()
        return sys.modules["torch"]

    def test_defaults_are_small(self) -> None:
        torch = self._configure()
        torch.set_num_threads.assert_called_once_with(4)
        torch.set_num_interop_threads.assert_called_once_with(1)

    def test_values_are_configurable(self) -> None:
        torch = self._configure(torch_num_threads=2, torch_num_interop_threads=3)
        torch.set_num_threads.assert_called_once_with(2)
        torch.set_num_interop_threads.assert_called_once_with(3)

    def test_zero_keeps_torch_defaults(self) -> None:
        torch = self._configure(torch_num_threads=0, torch_num_interop_threads=0)
        torch.set_num_threads.assert_not_called()
        torch.set_num_interop_threads.assert_not_called()

    def test_applies_only_once(self) -> None:
        self._configure()
        torch = self._configure()
        torch.set_num_threads.assert_called_once()

    def test_interop_pool_already_started_is_tolerated(self) -> None:
        sys.modules["torch"].set_num_interop_threads.side_effect = RuntimeError("cannot set after start")
        torch = self._configure()
        torch.set_num_threads.assert_called_once_with(4)

    def test_lanes_apply_the_limits_before_any_work(self) -> None:
        torch = sys.modules["torch"]
        seen: list[int] = []

        def work() -> None:
            seen.append(torch.set_num_threads.call_count)

        config = AnimaWorksConfig(gpu=GPUConfig())
        with patch("core.config.load_config", return_value=config):
            run_inference("embedding", "cpu", work)
        assert seen == [1]

    def test_broken_torch_does_not_break_the_lane(self) -> None:
        sys.modules["torch"].set_num_threads.side_effect = ValueError("broken")
        assert run_inference("embedding", "cpu", lambda: "still works") == "still works"
        assert run_inference("embedding", "cpu", lambda: "again") == "again"


class TestChromaNativeThreads:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOKIO_WORKER_THREADS", raising=False)
        monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)

    def _bound(self, limit: int) -> None:
        from core.memory.rag.store import _bound_chroma_native_threads

        config = AnimaWorksConfig(rag=RAGConfig(chroma_native_threads=limit))
        with patch("core.config.load_config", return_value=config), patch("os.cpu_count", return_value=32):
            _bound_chroma_native_threads()

    def test_sets_both_pools_with_the_default_limit(self) -> None:
        self._bound(RAGConfig().chroma_native_threads)
        assert os.environ["TOKIO_WORKER_THREADS"] == "4"
        assert os.environ["RAYON_NUM_THREADS"] == "4"

    def test_never_raises_above_the_cpu_count(self) -> None:
        self._bound(64)
        assert os.environ["TOKIO_WORKER_THREADS"] == "32"

    def test_zero_keeps_chroma_defaults(self) -> None:
        self._bound(0)
        assert "TOKIO_WORKER_THREADS" not in os.environ
        assert "RAYON_NUM_THREADS" not in os.environ

    def test_operator_values_win(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOKIO_WORKER_THREADS", "9")
        self._bound(4)
        assert os.environ["TOKIO_WORKER_THREADS"] == "9"
        assert os.environ["RAYON_NUM_THREADS"] == "4"

    def test_task_runners_do_not_inherit_the_limits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.runtime.task_runner_supervisor import TaskRunnerSupervisor

        monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
        monkeypatch.setenv("RAYON_NUM_THREADS", "4")
        shared = tmp_path / "shared"
        supervisor = TaskRunnerSupervisor.__new__(TaskRunnerSupervisor)
        supervisor.shared_dir = shared
        supervisor.anima_name = "rin"
        supervisor.anima_dir = tmp_path / "rin"
        supervisor.socket_path = tmp_path / "sock"
        supervisor.root_epoch = "epoch"
        env = supervisor._build_child_environment({}, attempt=1, display_lane="chat")
        assert "TOKIO_WORKER_THREADS" not in env
        assert "RAYON_NUM_THREADS" not in env


def test_http_clients_never_import_torch(tmp_path: Path) -> None:
    """Runners call the server for inference, so they must not load torch themselves."""
    script = """
import sys
from unittest.mock import MagicMock, patch

import core.infra.gpu
import core.infra.inference
import core.memory.rag.embedding as embedding
import core.memory.retrieval.reranker as reranker

client = MagicMock()
client.post.return_value.json.side_effect = [{"embeddings": [[0.5]]}, {"scores": [0.25, 0.75]}]
with patch.object(embedding, "_shared_http_client", return_value=client):
    assert embedding.generate_embeddings(["text"]) == [[0.5]]
    ranked = reranker.CrossEncoderReranker("model").rerank_sync("query", [{"content": "a"}, {"content": "b"}])
assert ranked[0]["ce_score"] == 0.75
loaded = sorted(name for name in sys.modules if name.split(".")[0] in {"torch", "sentence_transformers", "chromadb"})
assert not loaded, loaded
"""
    env = {
        **os.environ,
        "ANIMAWORKS_DATA_DIR": str(tmp_path),
        "ANIMAWORKS_EMBED_URL": "http://127.0.0.1:9/api/internal/embed",
        "ANIMAWORKS_RERANK_URL": "http://127.0.0.1:9/api/internal/rerank",
    }
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=110
    )
    assert result.returncode == 0, result.stderr[-2000:]
