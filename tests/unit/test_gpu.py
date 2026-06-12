from __future__ import annotations

from unittest.mock import patch

from core.config.schemas import AnimaWorksConfig, GPUConfig, RAGConfig
from core.gpu import reset_gpu_status_for_testing, resolve_device


def _config(
    *,
    use_gpu: bool = False,
    embedding_device: str = "auto",
    nli_device: str = "cpu",
    reranker_device: str = "cpu",
) -> AnimaWorksConfig:
    return AnimaWorksConfig(
        rag=RAGConfig(use_gpu=use_gpu),
        gpu=GPUConfig(
            embedding_device=embedding_device,  # type: ignore[arg-type]
            nli_device=nli_device,  # type: ignore[arg-type]
            reranker_device=reranker_device,  # type: ignore[arg-type]
        ),
    )


def test_resolve_embedding_auto_respects_legacy_rag_use_gpu() -> None:
    reset_gpu_status_for_testing()
    with (
        patch("core.config.load_config", return_value=_config(use_gpu=False)),
        patch("core.gpu._cuda_available_safely", return_value=True),
    ):
        assert resolve_device("embedding") == "cpu"

    with (
        patch("core.config.load_config", return_value=_config(use_gpu=True)),
        patch("core.gpu._cuda_available_safely", return_value=True),
    ):
        assert resolve_device("embedding") == "cuda"


def test_resolve_device_honors_explicit_cpu_and_cuda_availability() -> None:
    reset_gpu_status_for_testing()
    with (
        patch("core.config.load_config", return_value=_config(use_gpu=False, embedding_device="cuda")),
        patch("core.gpu._cuda_available_safely", return_value=False),
    ):
        assert resolve_device("embedding") == "cpu"

    with (
        patch("core.config.load_config", return_value=_config(nli_device="cuda")),
        patch("core.gpu._cuda_available_safely", return_value=True),
    ):
        assert resolve_device("nli") == "cuda"

    with (
        patch("core.config.load_config", return_value=_config(reranker_device="cpu")),
        patch("core.gpu._cuda_available_safely", return_value=True),
    ):
        assert resolve_device("reranker") == "cpu"


def test_resolve_auto_for_small_models_uses_cuda_only_when_available() -> None:
    reset_gpu_status_for_testing()
    with (
        patch("core.config.load_config", return_value=_config(nli_device="auto")),
        patch("core.gpu._cuda_available_safely", return_value=False),
    ):
        assert resolve_device("nli") == "cpu"

    with (
        patch("core.config.load_config", return_value=_config(reranker_device="auto")),
        patch("core.gpu._cuda_available_safely", return_value=True),
    ):
        assert resolve_device("reranker") == "cuda"
