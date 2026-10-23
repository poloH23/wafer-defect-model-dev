from src.evaluation.model_card import render_model_card

NAMES = ["Center", "Edge_Loc"]
SPEC = {
    "class_names": NAMES,
    "thresholds": [0.5, 0.75],
    "num_pixel_values": 3,
    "image_shape": (52, 52),
    "invalid_pixel_policy": "neighbor_mode",
    "provenance": {"model_name": "shufflenet_v2_x1_0", "mlflow_run_id": "run123", "training_git_commit": "aaa", "export_git_commit": "bbb", "split_digest": "ddd"},
}
VERIFICATION = {"samples": 3700, "max_abs_difference": 2.0e-6, "decision_mismatches": 0, "batch_max_abs_difference": 0.0}
LABEL = "wafer_shufflenet_v2_x1_0 (learning_rate=0.003)"
PER_CLASS = {name: {"precision": 0.99, "recall": 0.98, "f1": 0.985, "support": 1200, "ap": 0.999} for name in NAMES}
TEST_REPORT = {
    "n_test": 3700,
    "summaries": [
        {
            "candidate": LABEL,
            "seeds": [0, 1, 2],
            "test_errors_tuned": {"mean": 46.0, "std": 1.7},
            "test_errors_tuned_ci": {"errors": 46.0, "ci_low": 34.3, "ci_high": 58.7},
            "test_exact_match": {"mean": 0.9876, "std": 0.0005},
            "test_macro_f1": {"mean": 0.9932, "std": 0.0004},
            "test_macro_ap": {"mean": 0.9995, "std": 0.0},
        }
    ],
    "runs": {LABEL: {"0": {"tuned": {"exact_match": 0.9876, "macro_f1": 0.9930, "macro_ap": 0.9995, "per_class": PER_CLASS}}}},
}
BENCHMARK_ROW = {
    "model": "shufflenet_v2_x1_0",
    "parameters": 1_260_000,
    "flops": 250_000_000,
    "onnx_mb": 5.3,
    "gpu_latency": {"median_ms": 2.31, "p95_ms": 3.16},
    "gpu_throughput_img_s": 7336.0,
    "cpu_server": {"median_ms": 0.82, "p95_ms": 1.10},
    "edge_proxy": {"median_ms": 1.43, "p95_ms": 1.64},
}


def test_card_reports_every_section_from_its_source():
    card = render_model_card(SPEC, VERIFICATION, TEST_REPORT, LABEL, 0, BENCHMARK_ROW, "f" * 64)
    assert "| Edge_Loc | 0.75 |" in card  # thresholds from the model's metadata
    assert "46.0 ± 1.7 (95% interval 34.3 to 58.7)" in card  # configuration, from the test evaluation
    assert "| Delivered model, seed 0 | 46 | 98.76%" in card  # the delivered seed's own result
    assert "| Center | 1200 | 0.9900 | 0.9800 | 0.9850 |" in card
    assert "7,336 maps/s" in card and "0.82 ms" in card and "1.43 ms" in card
    assert "Maps whose decisions differ: 0" in card
    assert "ffffffff" in card and "run123" in card


def test_missing_reports_are_stated_not_invented():
    card = render_model_card(SPEC, VERIFICATION, None, None, 0, None, "f" * 64)
    assert "No test evaluation report was found" in card
    assert "No benchmark of this architecture was found" in card
