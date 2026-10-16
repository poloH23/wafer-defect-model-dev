"""
Latency and throughput of a model on the machine at hand, for the three deployment targets:

- cloud GPU:  PyTorch on CUDA, single-image latency and large-batch throughput
- CPU server: ONNX Runtime on CPU with one thread per physical core, single-image latency
- edge proxy: ONNX Runtime on CPU limited to a single thread, single-image latency

The edge numbers come from one desktop CPU core, several times faster than the ARM cores of typical
edge devices: they rank models relative to each other, they are not edge-device latencies.

GPU timings cover the model computation only: the input is already on the GPU, so host-to-device
transfer and preprocessing, which a deployed service adds, are not included.
"""

import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn


def summarize_timings(seconds: list[float]) -> dict[str, float]:
    """Median and 95th percentile in milliseconds. The median resists occasional outliers, p95 shows the tail."""
    milliseconds = np.asarray(seconds) * 1000
    return {"median_ms": float(np.median(milliseconds)), "p95_ms": float(np.percentile(milliseconds, 95))}


def export_onnx(model: nn.Module, path: str | Path, input_shape: tuple[int, ...]) -> Path:
    """
    Export a CPU model to ONNX with a dynamic batch dimension, using PyTorch's default (dynamo-based)
    exporter, which requires the onnxscript package.
    """
    model.eval()
    example = torch.zeros((2, *input_shape[1:]))
    torch.onnx.export(model, (example,), str(path), input_names=["input"], output_names=["logits"], dynamic_shapes=({0: torch.export.Dim("batch")},), external_data=False)
    return Path(path)


def onnx_max_abs_diff(model: nn.Module, path: str | Path, input_shape: tuple[int, ...]) -> float:
    """Largest difference between the ONNX Runtime and PyTorch outputs on the same random input."""
    x = np.random.default_rng(0).standard_normal(input_shape).astype(np.float32)
    model.eval()
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    actual = make_cpu_session(path, threads=1).run(None, {"input": x})[0]
    return float(np.abs(actual - expected).max())


def make_cpu_session(path: str | Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def time_onnx_cpu(path: str | Path, input_shape: tuple[int, ...], threads: int, warmup: int = 20, iterations: int = 200) -> list[float]:
    """Wall-clock seconds per inference with ONNX Runtime on CPU, after `warmup` untimed runs."""
    session = make_cpu_session(path, threads)
    feed = {"input": np.random.default_rng(0).standard_normal(input_shape).astype(np.float32)}
    for _ in range(warmup):
        session.run(None, feed)
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, feed)
        timings.append(time.perf_counter() - start)
    return timings


@torch.no_grad()
def time_torch_cuda(model: nn.Module, input_shape: tuple[int, ...], warmup: int = 50, iterations: int = 200) -> list[float]:
    """
    Seconds per forward pass on the GPU, measured with CUDA events. GPU work is asynchronous, so CPU
    timers would only measure how long it takes to queue the work; events measure the work itself.
    """
    device = torch.device("cuda")
    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(iterations):
        start.record()
        model(x)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end) / 1000.0)
    return timings


def cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def physical_cores() -> int:
    """Physical cores (not SMT threads): ONNX Runtime compute rarely benefits from the extra logical threads."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            pairs, physical_id = set(), None
            for line in f:
                key, _, value = line.partition(":")
                if key.strip() == "physical id":
                    physical_id = value.strip()
                elif key.strip() == "core id":
                    pairs.add((physical_id, value.strip()))
        if pairs:
            return len(pairs)
    except OSError:
        pass
    return os.cpu_count() or 1


def hardware_info() -> dict[str, str | int]:
    info: dict[str, str | int] = {
        "cpu": cpu_model_name(),
        "physical_cores": physical_cores(),
        "logical_cpus": os.cpu_count() or 0,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
