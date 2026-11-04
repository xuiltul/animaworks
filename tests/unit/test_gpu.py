from __future__ import annotations

import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from core.config.schemas import AnimaWorksConfig, GPUConfig, RAGConfig
from core.infra import gpu
from core.infra.gpu import (
    get_gpu_status,
    is_gpu_failure,
    record_component_device,
    reset_gpu_status_for_testing,
    resolve_device,
)


def _config(
    *,
    use_gpu: bool = False,
    embedding_device: str = "auto",
    reranker_device: str = "cpu",
) -> AnimaWorksConfig:
    return AnimaWorksConfig(
        rag=RAGConfig(use_gpu=use_gpu),
        gpu=GPUConfig(
            embedding_device=embedding_device,  # type: ignore[arg-type]
            reranker_device=reranker_device,  # type: ignore[arg-type]
        ),
    )


_REAL_MPS_PROBE = gpu._mps_available_safely


@pytest.fixture(autouse=True)
def _no_mps_unless_requested():
    """Keep these results the same on a Mac; tests that want MPS patch it back on."""
    reset_gpu_status_for_testing()
    with patch("core.infra.gpu._mps_available_safely", return_value=False):
        yield
    reset_gpu_status_for_testing()


def _resolve_with(component: str, *, cuda: bool, mps: bool, **config: str | bool) -> str:
    with (
        patch("core.config.load_config", return_value=_config(**config)),  # type: ignore[arg-type]
        patch("core.infra.gpu._cuda_available_safely", return_value=cuda),
        patch("core.infra.gpu._mps_available_safely", return_value=mps),
    ):
        return resolve_device(component)


def test_resolve_embedding_auto_respects_legacy_rag_use_gpu() -> None:
    reset_gpu_status_for_testing()
    with (
        patch("core.config.load_config", return_value=_config(use_gpu=False)),
        patch("core.infra.gpu._cuda_available_safely", return_value=True),
    ):
        assert resolve_device("embedding") == "cpu"

    with (
        patch("core.config.load_config", return_value=_config(use_gpu=True)),
        patch("core.infra.gpu._cuda_available_safely", return_value=True),
    ):
        assert resolve_device("embedding") == "cuda"


def test_resolve_device_honors_explicit_cpu_and_cuda_availability() -> None:
    reset_gpu_status_for_testing()
    with (
        patch("core.config.load_config", return_value=_config(use_gpu=False, embedding_device="cuda")),
        patch("core.infra.gpu._cuda_available_safely", return_value=False),
    ):
        assert resolve_device("embedding") == "cpu"

    with (
        patch("core.config.load_config", return_value=_config(reranker_device="cpu")),
        patch("core.infra.gpu._cuda_available_safely", return_value=True),
    ):
        assert resolve_device("reranker") == "cpu"


def test_resolve_auto_for_small_models_uses_cuda_only_when_available() -> None:
    reset_gpu_status_for_testing()
    with (
        patch("core.config.load_config", return_value=_config(reranker_device="auto")),
        patch("core.infra.gpu._cuda_available_safely", return_value=True),
    ):
        assert resolve_device("reranker") == "cuda"


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_prefers_cuda_then_mps_then_cpu(cuda: bool, mps: bool, expected: str) -> None:
    assert _resolve_with("embedding", cuda=cuda, mps=mps, use_gpu=True) == expected
    assert _resolve_with("reranker", cuda=cuda, mps=mps, reranker_device="auto") == expected


def test_embedding_auto_keeps_cpu_without_use_gpu_even_with_mps() -> None:
    assert _resolve_with("embedding", cuda=False, mps=True, use_gpu=False) == "cpu"


def test_default_config_stays_on_cpu_with_mps_available() -> None:
    assert _resolve_with("embedding", cuda=False, mps=True) == "cpu"
    assert _resolve_with("reranker", cuda=False, mps=True) == "cpu"


def test_schema_defaults_are_unchanged() -> None:
    assert GPUConfig().embedding_device == "auto"
    assert GPUConfig().reranker_device == "cpu"
    assert RAGConfig().use_gpu is False


def test_requested_cuda_uses_mps_when_only_mps_exists() -> None:
    assert _resolve_with("reranker", cuda=False, mps=True, reranker_device="cuda") == "mps"
    assert _resolve_with("embedding", cuda=False, mps=True, embedding_device="cuda") == "mps"


def test_requested_cuda_is_kept_when_available() -> None:
    assert _resolve_with("reranker", cuda=True, mps=True, reranker_device="cuda") == "cuda"


def test_requested_mps_is_honoured_and_replaced_when_unavailable() -> None:
    assert _resolve_with("reranker", cuda=True, mps=True, reranker_device="mps") == "mps"
    assert _resolve_with("reranker", cuda=True, mps=False, reranker_device="mps") == "cuda"
    assert _resolve_with("reranker", cuda=False, mps=False, reranker_device="mps") == "cpu"


def test_substitution_is_logged_once_per_component(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="core.infra.gpu"):
        for _ in range(5):
            _resolve_with("reranker", cuda=False, mps=True, reranker_device="cuda")
            _resolve_with("embedding", cuda=False, mps=True, embedding_device="cuda")
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "cuda requested for reranker but unavailable; using mps",
        "cuda requested for embedding but unavailable; using mps",
    ]


def test_mps_probe_is_cached_and_tolerates_a_broken_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", broken)
    assert _REAL_MPS_PROBE() is False

    reset_gpu_status_for_testing()
    working = types.ModuleType("torch")
    working.backends = MagicMock()  # type: ignore[attr-defined]
    working.backends.mps.is_built.return_value = True
    working.backends.mps.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", working)
    assert _REAL_MPS_PROBE() is True
    assert _REAL_MPS_PROBE() is True
    assert working.backends.mps.is_available.call_count == 1


@pytest.mark.parametrize(
    "message",
    [
        "MPS backend out of memory (MPS allocated: 1.18 GB, other allocations: 0 bytes, max allowed: 2.50 GB).",
        "Invalid buffer size: 12.50 GB",
        "Placeholder storage has not been allocated on MPS device!",
        "MPSNDArray dimensions must be positive",
        "Metal command buffer failed",
        "CUDA out of memory",
        "cudaGetDeviceCount Error 804",
    ],
)
def test_gpu_failures_are_recognized(message: str) -> None:
    assert is_gpu_failure(RuntimeError(message)) is True


def test_unsupported_mps_operator_is_recognized() -> None:
    message = "The operator 'aten::foo' is not currently implemented for the MPS device."
    assert is_gpu_failure(NotImplementedError(message)) is True


@pytest.mark.parametrize("message", ["timestamps are out of order", "the bumps and dumps failed", "shape mismatch"])
def test_ordinary_errors_are_not_gpu_failures(message: str) -> None:
    assert is_gpu_failure(RuntimeError(message)) is False


def test_non_runtime_errors_are_not_gpu_failures() -> None:
    assert is_gpu_failure(ValueError("MPS backend out of memory")) is False


def test_status_reports_mps() -> None:
    record_component_device("embedding", "mps")
    assert get_gpu_status()["embedding_device"] == "mps"
