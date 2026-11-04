from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest

from core.config.schemas import AnimaWorksConfig, GPUConfig


@pytest.fixture
def mock_sentence_transformers_cross_encoder():
    mock_module = types.ModuleType("sentence_transformers")
    cross_encoder = MagicMock()
    mock_module.CrossEncoder = cross_encoder
    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = mock_module
    yield cross_encoder
    if original is not None:
        sys.modules["sentence_transformers"] = original
    else:
        sys.modules.pop("sentence_transformers", None)


@pytest.fixture(autouse=True)
def _reset_gpu_state():
    from core.infra.gpu import reset_gpu_status_for_testing

    reset_gpu_status_for_testing()
    with patch("core.infra.gpu._mps_available_safely", return_value=False):
        yield
    reset_gpu_status_for_testing()


def test_reranker_passes_resolved_device_to_cross_encoder(
    mock_sentence_transformers_cross_encoder,
) -> None:
    model = MagicMock()
    mock_sentence_transformers_cross_encoder.return_value = model
    config = AnimaWorksConfig(gpu=GPUConfig(reranker_device="cpu"))

    with patch("core.config.load_config", return_value=config):
        from core.memory.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker("test-reranker")
        assert reranker._ensure_model() is True

    mock_sentence_transformers_cross_encoder.assert_called_once_with("test-reranker", device="cpu")


def test_reranker_scoring_hides_the_progress_bar(mock_sentence_transformers_cross_encoder) -> None:
    model = MagicMock()
    model.predict.return_value = [0.5]
    mock_sentence_transformers_cross_encoder.return_value = model
    config = AnimaWorksConfig(gpu=GPUConfig(reranker_device="cpu"))

    with patch("core.config.load_config", return_value=config):
        from core.memory.retrieval.reranker import CrossEncoderReranker

        assert CrossEncoderReranker("test-reranker").score_sync("query", ["doc"]) == [0.5]

    model.predict.assert_called_once_with([["query", "doc"]], show_progress_bar=False)


def test_reranker_cuda_inference_failure_falls_back_to_cpu_and_records_status(
    mock_sentence_transformers_cross_encoder,
) -> None:
    gpu_model = MagicMock()
    gpu_model.predict.side_effect = RuntimeError("CUDA device lost")
    cpu_model = MagicMock()
    cpu_model.predict.return_value = [0.75]
    mock_sentence_transformers_cross_encoder.side_effect = [gpu_model, cpu_model]
    config = AnimaWorksConfig(gpu=GPUConfig(reranker_device="cuda"))

    with (
        patch("core.config.load_config", return_value=config),
        patch("core.infra.gpu._cuda_available_safely", return_value=True),
    ):
        from core.infra.gpu import get_gpu_status
        from core.memory.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker("test-reranker")
        result = reranker.rerank_sync("query", [{"content": "doc"}], min_candidates=1)
        status = get_gpu_status()

    assert result[0]["ce_score"] == 0.75
    assert mock_sentence_transformers_cross_encoder.call_args_list == [
        call("test-reranker", device="cuda"),
        call("test-reranker", device="cpu"),
    ]
    assert status["degraded"] is True
    assert "CUDA device lost" in str(status["last_error"])


def test_concurrent_ensure_model_loads_once(
    mock_sentence_transformers_cross_encoder,
) -> None:
    """Concurrent first-use callers must share one CrossEncoder load.

    Regression test for the triple simultaneous "Loading weights" bursts
    observed on 2026-07-17: _ensure_model() had no lock, so racing threads
    each loaded (and leaked) their own model copy.
    """
    import threading
    import time

    def slow_cross_encoder(*args, **kwargs):
        time.sleep(0.05)
        return MagicMock()

    mock_sentence_transformers_cross_encoder.side_effect = slow_cross_encoder
    config = AnimaWorksConfig(gpu=GPUConfig(reranker_device="cpu"))

    with patch("core.config.load_config", return_value=config):
        from core.memory.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker("test-reranker")
        threads = [threading.Thread(target=reranker._ensure_model) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert mock_sentence_transformers_cross_encoder.call_count == 1


def test_get_reranker_returns_same_instance_across_threads(
    mock_sentence_transformers_cross_encoder,
) -> None:
    import threading

    config = AnimaWorksConfig(gpu=GPUConfig(reranker_device="cpu"))
    results: list[object] = []

    with patch("core.config.load_config", return_value=config):
        import core.memory.retrieval.reranker as reranker_mod

        reranker_mod._reranker = None
        threads = [
            threading.Thread(target=lambda: results.append(reranker_mod.get_reranker("test-r"))) for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        reranker_mod._reranker = None

    assert len(results) == 8
    assert all(r is results[0] for r in results)
