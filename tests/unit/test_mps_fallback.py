"""MPS placement and CPU fallback for embedding and reranking, on a mocked device layer."""

from __future__ import annotations

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from core.config.schemas import AnimaWorksConfig, GPUConfig, RAGConfig

MPS_OOM = "MPS backend out of memory (MPS allocated: 1.18 GB, other allocations: 0 bytes, max allowed: 2.50 GB)."


def _model_on(device_type: str) -> MagicMock:
    model = MagicMock()
    model.device.type = device_type
    return model


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)
    monkeypatch.delenv("ANIMAWORKS_EMBED_URL", raising=False)
    monkeypatch.delenv("ANIMAWORKS_RERANK_URL", raising=False)
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    from tests.helpers.rag import reset_rag_state

    reset_rag_state()
    torch = types.ModuleType("torch")
    torch.set_num_threads = MagicMock()  # type: ignore[attr-defined]
    torch.set_num_interop_threads = MagicMock()  # type: ignore[attr-defined]
    torch.get_num_threads = MagicMock(return_value=4)  # type: ignore[attr-defined]
    torch.get_num_interop_threads = MagicMock(return_value=1)  # type: ignore[attr-defined]
    torch.mps = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    with (
        patch("core.infra.gpu._cuda_available_safely", return_value=False),
        patch("core.infra.gpu._mps_available_safely", return_value=True),
    ):
        yield torch
    reset_rag_state()


@pytest.fixture
def sentence_transformers(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = MagicMock()  # type: ignore[attr-defined]
    module.CrossEncoder = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


@pytest.fixture
def config():
    value = AnimaWorksConfig(
        rag=RAGConfig(use_gpu=True, embedding_model="model-x"),
        gpu=GPUConfig(reranker_device="cuda", embedding_batch_size=8),
    )
    with patch("core.config.load_config", return_value=value):
        yield value


def _status() -> dict[str, object]:
    from core.infra.gpu import get_gpu_status

    return get_gpu_status()


class TestEmbeddingOnMps:
    def test_loads_on_mps_and_reports_it(self, sentence_transformers, config, tmp_path) -> None:
        from core.memory.rag.embedding import get_embedding_model

        get_embedding_model()
        sentence_transformers.SentenceTransformer.assert_called_once_with(
            "model-x", cache_folder=str(tmp_path / "models"), device="mps"
        )
        assert _status()["embedding_device"] == "mps"
        assert _status()["degraded"] is False

    def test_load_failure_falls_back_to_cpu(self, sentence_transformers, config, tmp_path) -> None:
        from core.memory.rag.embedding import get_embedding_model

        cpu_model = _model_on("cpu")
        sentence_transformers.SentenceTransformer.side_effect = [RuntimeError(MPS_OOM), cpu_model]
        assert get_embedding_model() is cpu_model
        assert [item.kwargs["device"] for item in sentence_transformers.SentenceTransformer.call_args_list] == [
            "mps",
            "cpu",
        ]
        assert _status()["embedding_device"] == "cpu"
        assert _status()["degraded"] is True

    def test_out_of_memory_falls_back_to_cpu_and_clears_the_cache(
        self, sentence_transformers, config, tmp_path, _isolated
    ) -> None:
        from core.memory.rag.embedding import thread_safe_encode

        mps_model = _model_on("mps")
        mps_model.encode.side_effect = RuntimeError(MPS_OOM)
        cpu_model = _model_on("cpu")
        cpu_model.encode.return_value = np.array([[0.5, 0.25]])
        sentence_transformers.SentenceTransformer.side_effect = [mps_model, cpu_model]

        assert thread_safe_encode(["hello"]) == [[0.5, 0.25]]

        assert sentence_transformers.SentenceTransformer.call_args_list == [
            call("model-x", cache_folder=str(tmp_path / "models"), device="mps"),
            call("model-x", cache_folder=str(tmp_path / "models"), device="cpu"),
        ]
        assert _status()["degraded"] is True
        assert "MPS backend out of memory" in str(_status()["last_error"])
        _isolated.mps.empty_cache.assert_called_once()

    def test_degraded_process_stays_on_cpu(self, sentence_transformers, config) -> None:
        from core.memory.rag.embedding import get_embedding_model, thread_safe_encode

        mps_model = _model_on("mps")
        mps_model.encode.side_effect = RuntimeError(MPS_OOM)
        cpu_model = _model_on("cpu")
        cpu_model.encode.return_value = np.array([[0.5]])
        sentence_transformers.SentenceTransformer.side_effect = [mps_model, cpu_model]

        thread_safe_encode(["hello"])
        thread_safe_encode(["again"])

        assert get_embedding_model() is cpu_model
        mps_model.encode.assert_called_once()
        assert sentence_transformers.SentenceTransformer.call_count == 2

    def test_callers_failing_together_load_one_cpu_model(self, sentence_transformers, config) -> None:
        from core.memory.rag import embedding

        mps_model = _model_on("mps")
        cpu_model = _model_on("cpu")
        sentence_transformers.SentenceTransformer.side_effect = [mps_model, cpu_model]
        embedding.get_embedding_model()

        assert embedding._reload_embedding_model_on_device("model-x", "cpu") is cpu_model
        assert embedding._reload_embedding_model_on_device("model-x", "cpu") is cpu_model
        assert sentence_transformers.SentenceTransformer.call_count == 2

    def test_other_errors_do_not_fall_back(self, sentence_transformers, config) -> None:
        from core.memory.rag.embedding import thread_safe_encode

        mps_model = _model_on("mps")
        mps_model.encode.side_effect = RuntimeError("shape mismatch")
        sentence_transformers.SentenceTransformer.return_value = mps_model

        with pytest.raises(RuntimeError, match="shape mismatch"):
            thread_safe_encode(["hello"])
        assert _status()["degraded"] is False

    def test_stale_mps_model_stays_on_the_mps_thread(self, sentence_transformers, config) -> None:
        """A caller holding the MPS model after a fallback must not run it on the CPU lane."""
        from core.memory.rag.embedding import _encode_on_inference_thread

        threads: dict[str, str] = {}

        def record(device: str):
            def encode(*_args, **_kwargs):
                threads[device] = threading.current_thread().name
                return np.array([[0.0]])

            return encode

        mps_model, cpu_model = _model_on("mps"), _model_on("cpu")
        mps_model.encode.side_effect = record("mps")
        cpu_model.encode.side_effect = record("cpu")
        _encode_on_inference_thread(mps_model, ["a"])
        _encode_on_inference_thread(cpu_model, ["a"])
        assert threads["mps"].startswith("inference-mps")
        assert threads["cpu"].startswith("inference-embedding")

    def test_model_load_and_encode_stay_off_the_callers_thread(self, sentence_transformers, config) -> None:
        from core.memory.rag.embedding import thread_safe_encode

        names: list[str] = []
        model = _model_on("mps")

        def construct(*_args, **_kwargs):
            names.append(threading.current_thread().name)
            return model

        def encode(*_args, **_kwargs):
            names.append(threading.current_thread().name)
            return np.array([[0.0]])

        sentence_transformers.SentenceTransformer.side_effect = construct
        model.encode.side_effect = encode

        with ThreadPoolExecutor(max_workers=4) as callers:
            list(callers.map(lambda _: thread_safe_encode(["a"]), range(8)))
        assert {name.rsplit("_", 1)[0] for name in names} == {"inference-mps"}


class TestRerankerOnMps:
    def _reranker(self):
        from core.memory.retrieval.reranker import CrossEncoderReranker

        return CrossEncoderReranker("reranker-x")

    def test_live_config_requesting_cuda_runs_on_mps(self, sentence_transformers, config) -> None:
        model = _model_on("mps")
        model.predict.return_value = [0.5]
        sentence_transformers.CrossEncoder.return_value = model

        assert self._reranker().score_sync("query", ["doc"]) == [0.5]
        sentence_transformers.CrossEncoder.assert_called_once_with("reranker-x", device="mps")

    def test_load_failure_falls_back_to_cpu(self, sentence_transformers, config) -> None:
        cpu_model = _model_on("cpu")
        cpu_model.predict.return_value = [0.5]
        sentence_transformers.CrossEncoder.side_effect = [RuntimeError(MPS_OOM), cpu_model]

        assert self._reranker().score_sync("query", ["doc"]) == [0.5]
        assert sentence_transformers.CrossEncoder.call_args_list == [
            call("reranker-x", device="mps"),
            call("reranker-x", device="cpu"),
        ]

    def test_out_of_memory_falls_back_to_cpu_and_clears_the_cache(
        self, sentence_transformers, config, _isolated
    ) -> None:
        mps_model = _model_on("mps")
        mps_model.predict.side_effect = RuntimeError(MPS_OOM)
        cpu_model = _model_on("cpu")
        cpu_model.predict.return_value = [0.75]
        sentence_transformers.CrossEncoder.side_effect = [mps_model, cpu_model]
        reranker = self._reranker()

        assert reranker.score_sync("query", ["doc"]) == [0.75]
        assert reranker.score_sync("query", ["doc"]) == [0.75]

        cpu_model.predict.assert_called_with([["query", "doc"]], show_progress_bar=False)
        mps_model.predict.assert_called_once()
        assert sentence_transformers.CrossEncoder.call_count == 2
        assert _status()["degraded"] is True
        _isolated.mps.empty_cache.assert_called_once()

    def test_other_errors_skip_rerank_without_falling_back(self, sentence_transformers, config) -> None:
        model = _model_on("mps")
        model.predict.side_effect = RuntimeError("shape mismatch")
        sentence_transformers.CrossEncoder.return_value = model

        assert self._reranker().score_sync("query", ["doc"]) is None
        sentence_transformers.CrossEncoder.assert_called_once()
        assert _status()["degraded"] is False

    def test_embedding_and_reranking_share_the_mps_thread(self, sentence_transformers, config) -> None:
        from core.memory.rag.embedding import thread_safe_encode

        names: set[str] = set()

        def embed(*_args, **_kwargs):
            names.add(threading.current_thread().name)
            return np.array([[0.0]])

        def predict(*_args, **_kwargs):
            names.add(threading.current_thread().name)
            return [0.5]

        embedder = _model_on("mps")
        embedder.encode.side_effect = embed
        reranker_model = _model_on("mps")
        reranker_model.predict.side_effect = predict
        sentence_transformers.SentenceTransformer.return_value = embedder
        sentence_transformers.CrossEncoder.return_value = reranker_model

        thread_safe_encode(["a"])
        self._reranker().score_sync("query", ["doc"])
        assert len(names) == 1
        assert names.pop().startswith("inference-mps")
