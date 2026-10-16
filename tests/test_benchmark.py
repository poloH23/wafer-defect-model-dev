from pathlib import Path

import pytest
import torch

from src.models.benchmark import export_onnx, hardware_info, onnx_max_abs_diff, physical_cores, summarize_timings, time_onnx_cpu, time_torch_cuda
from src.models.factory import build_model
from src.models.profiling import count_parameters

INPUT_SHAPE = (1, 3, 52, 52)


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, Path]:
    model = build_model("mobilenet_v3_small", num_classes=8, in_channels=3)
    return model, export_onnx(model, tmp_path_factory.mktemp("onnx") / "model.onnx", INPUT_SHAPE)


def test_summarize_timings_in_milliseconds():
    summary = summarize_timings([0.001, 0.002, 0.003])
    assert summary["median_ms"] == pytest.approx(2.0)
    assert summary["p95_ms"] == pytest.approx(2.9)


def test_export_is_a_single_self_contained_file(exported):
    model, path = exported
    assert sorted(p.name for p in path.parent.iterdir()) == ["model.onnx"]  # no separate .onnx.data file
    assert path.stat().st_size > 4 * count_parameters(model)  # FP32 weights are inside


@pytest.mark.parametrize("batch", [1, 5])
def test_onnx_matches_pytorch_for_any_batch_size(exported, batch):
    model, path = exported
    assert onnx_max_abs_diff(model, path, (batch, *INPUT_SHAPE[1:])) < 1e-4


def test_cpu_timing(exported):
    _, path = exported
    timings = time_onnx_cpu(path, INPUT_SHAPE, threads=1, warmup=2, iterations=5)
    assert len(timings) == 5 and all(t > 0 for t in timings)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_gpu_timing():
    model = build_model("mobilenet_v3_small", num_classes=8, in_channels=3)
    timings = time_torch_cuda(model, INPUT_SHAPE, warmup=2, iterations=5)
    assert len(timings) == 5 and all(t > 0 for t in timings)


def test_hardware_info():
    info = hardware_info()
    assert physical_cores() >= 1
    assert {"cpu", "physical_cores", "logical_cpus", "torch", "onnxruntime"} <= set(info)
