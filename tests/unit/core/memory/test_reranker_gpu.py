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
    from core.gpu import reset_gpu_status_for_testing

    reset_gpu_status_for_testing()
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
        patch("core.gpu._cuda_available_safely", return_value=True),
    ):
        from core.gpu import get_gpu_status
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
