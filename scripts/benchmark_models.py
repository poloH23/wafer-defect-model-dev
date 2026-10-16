"""
Measure size, compute, latency and throughput of model architectures on this machine, for the three
deployment targets (cloud GPU, CPU server, edge proxy; see src/models/benchmark.py).

Speed depends only on the architecture and the input size, not on the trained weights, so models are
built with random weights from conf/model/<name>.yaml. Run it on an otherwise idle machine:
a training job running at the same time distorts every timing.

Usage:
    python scripts/benchmark_models.py --dataset wafer_defect --models resnet18 mobilenet_v3_small \\
        --output reports/experiments/benchmark.md
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.common.config_schema import load_dataset_config, load_model_config, resolve_path
from src.data.transforms import encoded_channels
from src.models.benchmark import export_onnx, hardware_info, onnx_max_abs_diff, physical_cores, summarize_timings, time_onnx_cpu, time_torch_cuda
from src.models.factory import build_model
from src.models.profiling import count_flops, count_parameters

# An ONNX export whose outputs differ from PyTorch by more than this is treated as broken
MAX_EXPORT_DIFF = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model architectures for the three deployment targets.")
    parser.add_argument("--dataset", required=True, help="Dataset config name (conf/dataset/<name>.yaml), for the input shape and number of classes.")
    parser.add_argument("--models", nargs="+", required=True, help="Model config names (conf/model/<name>.yaml).")
    parser.add_argument("--encoding", default="onehot", choices=["onehot", "scaled"], help="Input encoding, which sets the number of input channels.")
    parser.add_argument("--server-threads", type=int, default=physical_cores(), help="CPU threads for the CPU-server target (default: physical cores).")
    parser.add_argument("--edge-threads", type=int, default=1, help="CPU threads for the edge proxy.")
    parser.add_argument("--gpu-batch", type=int, default=256, help="Batch size for GPU throughput.")
    parser.add_argument("--output", help="Optional Markdown report; a JSON file with every measurement is written next to it.")
    return parser.parse_args()


def benchmark_model(name: str, num_classes: int, input_shape: tuple[int, ...], args: argparse.Namespace, tmp_dir: Path) -> dict[str, Any]:
    model_cfg = load_model_config(name)
    model = build_model(model_cfg.name, num_classes, input_shape[1], pretrained=False, small_input_stem=model_cfg.small_input_stem).eval()
    result: dict[str, Any] = {"model": name, "parameters": count_parameters(model), "flops": count_flops(model, input_shape)}

    onnx_path = export_onnx(model, tmp_dir / f"{name}.onnx", input_shape)
    diff = onnx_max_abs_diff(model, onnx_path, input_shape)
    if diff > MAX_EXPORT_DIFF:
        raise RuntimeError(f"{name}: ONNX output differs from PyTorch by {diff:.2e}")
    result["onnx_mb"] = onnx_path.stat().st_size / 1e6
    result["onnx_max_abs_diff"] = diff

    result["cpu_server"] = summarize_timings(time_onnx_cpu(onnx_path, input_shape, threads=args.server_threads))
    result["edge_proxy"] = summarize_timings(time_onnx_cpu(onnx_path, input_shape, threads=args.edge_threads))

    if torch.cuda.is_available():
        result["gpu_latency"] = summarize_timings(time_torch_cuda(model, input_shape))
        batch_shape = (args.gpu_batch, *input_shape[1:])
        batch_timing = summarize_timings(time_torch_cuda(model, batch_shape, warmup=20, iterations=50))
        result["gpu_throughput_img_s"] = args.gpu_batch / (batch_timing["median_ms"] / 1000)
        model.cpu()
    return result


def to_markdown(results: list[dict[str, Any]], hardware: dict[str, Any], args: argparse.Namespace) -> str:
    def latency(entry: dict[str, float] | None) -> str:
        return "–" if entry is None else f"{entry['median_ms']:.2f} ({entry['p95_ms']:.2f})"

    lines = [
        f"Hardware: GPU {hardware.get('gpu', 'none')}; CPU {hardware['cpu']} ({hardware['physical_cores']} physical cores); "
        f"torch {hardware['torch']}, onnxruntime {hardware['onnxruntime']}.",
        "",
        f"Single-image latency in ms, as median (p95). GPU: PyTorch FP32, input already on the GPU. "
        f"CPU server: ONNX Runtime, {args.server_threads} threads. Edge proxy: ONNX Runtime, {args.edge_threads} thread(s), "
        "only meaningful to rank models, not as an edge-device latency.",
        "",
        f"| Model | Params (M) | MFLOPs | ONNX (MB) | GPU latency | GPU batch {args.gpu_batch} (img/s) | CPU server latency | Edge proxy latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        throughput = f"{r['gpu_throughput_img_s']:,.0f}" if "gpu_throughput_img_s" in r else "–"
        lines.append(
            f"| {r['model']} | {r['parameters'] / 1e6:.2f} | {r['flops'] / 1e6:,.0f} | {r['onnx_mb']:.1f} | "
            f"{latency(r.get('gpu_latency'))} | {throughput} | {latency(r['cpu_server'])} | {latency(r['edge_proxy'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = True  # let cuDNN pick the fastest kernels, as a deployed service would
    dataset_cfg = load_dataset_config(args.dataset)
    if dataset_cfg.num_pixel_values is None:
        raise ValueError("Set num_pixel_values in the dataset config.")
    with np.load(dataset_cfg.source_path) as data:
        image_shape = data[dataset_cfg.image_key].shape[1:]
    input_shape = (1, encoded_channels(dataset_cfg.num_pixel_values, args.encoding), *image_shape)

    hardware = hardware_info()
    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name in args.models:
            print(f"Benchmarking {name}...")
            results.append(benchmark_model(name, dataset_cfg.num_classes, input_shape, args, Path(tmp_dir)))

    table = to_markdown(results, hardware, args)
    print(table)
    if args.output:
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table, encoding="utf-8")
        details = {"hardware": hardware, "input_shape": list(input_shape), "settings": vars(args), "results": results}
        output.with_suffix(".json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        print(f"Saved: {Path(args.output).as_posix()} and {Path(args.output).with_suffix('.json').as_posix()}")


if __name__ == "__main__":
    main()
