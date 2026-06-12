"""CUDA fallback tests for core/memory/rag/singleton.py."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest

from core.config.schemas import AnimaWorksConfig, GPUConfig, RAGConfig


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)
    monkeypatch.delenv("ANIMAWORKS_EMBED_URL", raising=False)

    from core.memory.rag.singleton import _reset_for_testing

    _reset_for_testing()
    yield
    _reset_for_testing()


@pytest.fixture
def mock_sentence_transformers():
    mock_module = types.ModuleType("sentence_transformers")
    mock_class = MagicMock()
    mock_module.SentenceTransformer = mock_class
    sys.modules["sentence_transformers"] = mock_module
    yield mock_class
    sys.modules.pop("sentence_transformers", None)


def _install_mock_torch(monkeypatch, *, available: bool = True, device_count=1) -> None:
    """Install a minimal torch mock for CUDA availability checks."""
    mock_torch = types.ModuleType("torch")
    mock_torch.cuda = MagicMock()
    mock_torch.cuda.is_available.return_value = available
    if isinstance(device_count, Exception):
        mock_torch.cuda.device_count.side_effect = device_count
    else:
        mock_torch.cuda.device_count.return_value = device_count
    monkeypatch.setitem(sys.modules, "torch", mock_torch)


def test_cuda_initialization_failure_falls_back_to_cpu(tmp_path, monkeypatch, mock_sentence_transformers):
    """CUDA availability errors should not prevent local embedding use."""
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    _install_mock_torch(monkeypatch)

    config = MagicMock()
    config.rag.use_gpu = True
    mock_model = MagicMock()
    mock_sentence_transformers.side_effect = [
        RuntimeError("cudaGetDeviceCount Error 804"),
        mock_model,
    ]

    with patch("core.config.load_config", return_value=config):
        from core.memory.rag.singleton import get_embedding_model

        result = get_embedding_model("model-x")

    assert result is mock_model
    assert mock_sentence_transformers.call_args_list == [
        call("model-x", cache_folder=str(tmp_path / "models"), device="cuda"),
        call("model-x", cache_folder=str(tmp_path / "models"), device="cpu"),
    ]


def test_cuda_device_count_failure_uses_cpu_without_cuda_constructor(tmp_path, monkeypatch, mock_sentence_transformers):
    """Broken CUDA probing should route embeddings to CPU before model load."""
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    _install_mock_torch(monkeypatch, device_count=RuntimeError("cudaGetDeviceCount Error 804"))

    config = MagicMock()
    config.rag.use_gpu = True
    mock_model = MagicMock()
    mock_sentence_transformers.return_value = mock_model

    with patch("core.config.load_config", return_value=config):
        from core.memory.rag.singleton import get_embedding_model

        result = get_embedding_model("model-x")

    assert result is mock_model
    mock_sentence_transformers.assert_called_once_with(
        "model-x",
        cache_folder=str(tmp_path / "models"),
        device="cpu",
    )


def test_runtime_cuda_encode_failure_falls_back_to_cpu_and_records_status(
    tmp_path,
    monkeypatch,
    mock_sentence_transformers,
):
    """CUDA failures during encode should degrade embedding to CPU and retry once."""
    import numpy as np

    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    _install_mock_torch(monkeypatch)

    config = AnimaWorksConfig(
        rag=RAGConfig(use_gpu=True, embedding_model="model-x"),
        gpu=GPUConfig(embedding_batch_size=7),
    )
    gpu_model = MagicMock()
    gpu_model.encode.side_effect = RuntimeError("CUDA device lost")
    cpu_model = MagicMock()
    cpu_model.encode.return_value = np.array([[0.5, 0.25]])
    mock_sentence_transformers.side_effect = [gpu_model, cpu_model]

    with patch("core.config.load_config", return_value=config):
        from core.gpu import get_gpu_status
        from core.memory.rag.singleton import thread_safe_encode

        result = thread_safe_encode(["hello"])
        status = get_gpu_status()

    assert result == [[0.5, 0.25]]
    assert mock_sentence_transformers.call_args_list == [
        call("model-x", cache_folder=str(tmp_path / "models"), device="cuda"),
        call("model-x", cache_folder=str(tmp_path / "models"), device="cpu"),
    ]
    gpu_model.encode.assert_called_once_with(
        ["hello"],
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=7,
    )
    cpu_model.encode.assert_called_once_with(
        ["hello"],
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=7,
    )
    assert status["embedding_device"] == "cpu"
    assert status["degraded"] is True
    assert "CUDA device lost" in str(status["last_error"])
