from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest

from core.config.schemas import AnimaWorksConfig, GPUConfig


@pytest.fixture
def mock_transformers_pipeline():
    mock_module = types.ModuleType("transformers")
    pipeline = MagicMock()
    mock_module.pipeline = pipeline
    original = sys.modules.get("transformers")
    sys.modules["transformers"] = mock_module
    yield pipeline
    if original is not None:
        sys.modules["transformers"] = original
    else:
        sys.modules.pop("transformers", None)


@pytest.fixture(autouse=True)
def _reset_nli_and_gpu_state():
    from core.gpu import reset_gpu_status_for_testing
    from core.memory.nli import _reset_for_testing

    reset_gpu_status_for_testing()
    _reset_for_testing()
    yield
    reset_gpu_status_for_testing()
    _reset_for_testing()


def test_nli_pipeline_reused_across_contradiction_detectors(
    tmp_path,
    mock_transformers_pipeline,
) -> None:
    pipe = MagicMock(return_value=[{"label": "CONTRADICTION", "score": 0.91}])
    mock_transformers_pipeline.return_value = pipe
    config = AnimaWorksConfig(gpu=GPUConfig(nli_device="cpu"))

    with patch("core.config.load_config", return_value=config):
        from core.memory.contradiction import ContradictionDetector

        detector_a = ContradictionDetector(tmp_path, "alice")
        detector_b = ContradictionDetector(tmp_path, "bob")

        assert detector_a._get_nli_model().check("hypothesis", "premise") == ("contradiction", 0.91)
        assert detector_b._get_nli_model().check("other", "premise") == ("contradiction", 0.91)

    mock_transformers_pipeline.assert_called_once_with(
        "text-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        device=-1,
    )
    assert pipe.call_count == 2


def test_nli_cuda_inference_failure_falls_back_to_cpu_and_records_status(
    mock_transformers_pipeline,
) -> None:
    gpu_pipe = MagicMock(side_effect=RuntimeError("CUDA device lost"))
    cpu_pipe = MagicMock(return_value=[{"label": "ENTAILMENT", "score": 0.88}])
    mock_transformers_pipeline.side_effect = [gpu_pipe, cpu_pipe]
    config = AnimaWorksConfig(gpu=GPUConfig(nli_device="cuda"))

    with (
        patch("core.config.load_config", return_value=config),
        patch("core.gpu._cuda_available_safely", return_value=True),
    ):
        from core.gpu import get_gpu_status
        from core.memory.nli import SharedNLIModel

        model = SharedNLIModel()
        assert model.check("hypothesis", "premise") == ("entailment", 0.88)
        status = get_gpu_status()

    assert mock_transformers_pipeline.call_args_list == [
        call(
            "text-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            device=0,
        ),
        call(
            "text-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            device=-1,
        ),
    ]
    assert status["degraded"] is True
    assert "CUDA device lost" in str(status["last_error"])
