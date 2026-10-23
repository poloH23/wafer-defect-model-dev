"""
A model card for the exported classifier: what the model is for, its input and output contract,
how well it performs, how fast it is, what it does not cover, and where it comes from.
Every number is read from the reports that produced it, never copied by hand.
"""

from typing import Any


def _fmt(stats: dict[str, float], digits: int = 1, percent: bool = False) -> str:
    scale, unit = (100, "%") if percent else (1, "")
    return f"{stats['mean'] * scale:.{digits}f}{unit} ± {stats['std'] * scale:.{digits}f}{unit}"


def render_model_card(
    spec: dict[str, Any],
    verification: dict[str, Any],
    test_report: dict[str, Any] | None,
    selected_label: str | None,
    deliverable_seed: int,
    benchmark_row: dict[str, Any] | None,
    onnx_sha256: str,
) -> str:
    """
    spec: class_names, thresholds, image_shape, num_pixel_values, invalid_pixel_policy, provenance.
    verification: the export checks against PyTorch on the validation split.
    test_report / benchmark_row: the test evaluation and the benchmark of the selected model, if available.
    """
    provenance = spec["provenance"]
    height, width = spec["image_shape"]
    lines = [
        "# Model card: wafer defect classifier",
        "",
        "## Model",
        "",
        f"- **Architecture**: {provenance['model_name']}, trained from scratch on MixedWM38 with rotation and flip augmentation.",
        f"- **Delivered model**: seed {deliverable_seed} of the configuration selected on the validation split for all three",
        "  deployment targets (cloud GPU, CPU server, edge), following docs/model_selection.md.",
        "- **Format**: a single ONNX file; the input encoding and the sigmoid are part of the graph, the thresholds and class",
        "  names are in its metadata. Reference inference with NumPy and ONNX Runtime only: src/inference/classifier.py.",
        "",
        "## Intended use",
        "",
        f"Multi-label classification of {height}x{width} wafer maps into {len(spec['class_names'])} defect types, any combination of",
        "which may be present. Trained and evaluated on a partly synthetic public dataset, not on production data: it supports",
        "the analysis of wafer maps, but its decisions should be validated on the target production line before being relied on.",
        "",
        "## Input and output",
        "",
        f"- **Input** 'wafer_map': uint8, shape (N, {height}, {width}); one category id per die, 0 outside the wafer, 1 pass,",
        f"  2 fail. Values of {spec['num_pixel_values']} or more are handled as in training ({spec['invalid_pixel_policy']}).",
        f"- **Output** 'probabilities': float32, shape (N, {len(spec['class_names'])}), one probability per class, in the order below.",
        "- **Decision**: a class is present when its probability reaches its threshold, tuned on the validation split.",
        "",
        "| Class | Threshold |",
        "|---|---|",
        *[f"| {name} | {threshold:.2f} |" for name, threshold in zip(spec["class_names"], spec["thresholds"], strict=True)],
        "",
        "## Performance",
        "",
    ]
    if test_report is None or selected_label is None:
        lines += ["No test evaluation report was found; performance is not documented here.", ""]
    else:
        summary = next(s for s in test_report["summaries"] if s["candidate"] == selected_label)
        delivered = test_report["runs"][selected_label][str(deliverable_seed)]["tuned"]
        n_test = test_report["n_test"]
        ci = summary["test_errors_tuned_ci"]
        lines += [
            f"Evaluated once on the held-out test split ({n_test:,} maps), with per-class thresholds tuned on the validation split.",
            "Errors are maps with any label wrong.",
            "",
            "| | Errors | Exact match | Macro F1 | Macro average precision |",
            "|---|---|---|---|---|",
            f"| Configuration, {len(summary['seeds'])} seeds | {_fmt(summary['test_errors_tuned'])} (95% interval {ci['ci_low']:.1f} to {ci['ci_high']:.1f}) "
            f"| {_fmt(summary['test_exact_match'], 2, percent=True)} | {_fmt(summary['test_macro_f1'], 4)} | {_fmt(summary['test_macro_ap'], 4)} |",
            f"| Delivered model, seed {deliverable_seed} | {(1 - delivered['exact_match']) * n_test:.0f} | {delivered['exact_match']:.2%} "
            f"| {delivered['macro_f1']:.4f} | {delivered['macro_ap']:.4f} |",
            "",
            f"Per class, delivered model (seed {deliverable_seed}):",
            "",
            "| Class | Test maps | Precision | Recall | F1 |",
            "|---|---|---|---|---|",
            *[
                f"| {name} | {m['support']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
                for name, m in ((name, delivered["per_class"][name]) for name in spec["class_names"])
            ],
            "",
        ]
    lines += ["## Efficiency", ""]
    if benchmark_row is None:
        lines += ["No benchmark of this architecture was found; efficiency is not documented here.", ""]
    else:
        gpu = benchmark_row.get("gpu_latency")
        lines += [
            f"{benchmark_row['parameters'] / 1e6:.2f}M parameters, {benchmark_row['flops'] / 1e6:,.0f} MFLOPs per map, "
            f"{benchmark_row['onnx_mb']:.1f} MB as ONNX. Median single-map latency (reports/experiments/benchmark.md):",
            "",
            f"- GPU (PyTorch FP32): {gpu['median_ms']:.2f} ms; {benchmark_row['gpu_throughput_img_s']:,.0f} maps/s in batches" if gpu else "- GPU: not measured",
            f"- CPU server (ONNX Runtime, multiple threads): {benchmark_row['cpu_server']['median_ms']:.2f} ms",
            f"- Edge proxy (ONNX Runtime, 1 desktop thread; ranks models, not a device latency): {benchmark_row['edge_proxy']['median_ms']:.2f} ms",
            "",
        ]
    lines += [
        "## Export verification",
        "",
        f"On the {verification['samples']:,} maps of the validation split, raw maps through the ONNX file and the reference",
        "inference against the PyTorch checkpoint on the training pipeline's inputs:",
        "",
        f"- Largest probability difference: {verification['max_abs_difference']:.1e}",
        f"- Maps whose decisions differ: {verification['decision_mismatches']}",
        f"- Largest difference between batched and one-by-one inference: {verification['batch_max_abs_difference']:.1e}",
        "",
        "## Limitations",
        "",
        "- Partly synthetic training data with an artificial class balance: precision on a production line will differ.",
        "- Ambiguous labels between Edge_Loc and Edge_Ring: errors concentrate where the edge fails almost all around.",
        "- Rare classes (Near_Full, Random) have few test maps, so their per-class figures are imprecise.",
        "- See the Limitations section of the README for the full list, including those of the model selection.",
        "",
        "## Provenance",
        "",
        f"- MLflow run: {provenance['mlflow_run_id']}",
        f"- Training code commit: {provenance['training_git_commit']}; export code commit: {provenance['export_git_commit']}",
        f"- Data split digest: {provenance['split_digest']}",
        f"- ONNX file SHA-256: {onnx_sha256}",
        "",
    ]
    return "\n".join(lines)
