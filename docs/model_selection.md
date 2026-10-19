# Model Selection Protocol

This document fixes how the final model of each deployment target is chosen. It is committed before
the custom-architecture search and before any evaluation on the test split, so that the choice
cannot be adapted to results seen afterwards. The commit history is the record of that order.

## Evidence used

All decisions use the **validation split of split `v2`** only. The test split is evaluated once per
selected model, after this protocol has been applied, and nothing below may change after that.

- **Accuracy**: number of misclassified validation samples (any of the 8 labels wrong), at a decision
  threshold of 0.5. Each run is represented by the mean of its last 5 epochs; each candidate by the
  mean over 3 seeds, trained for the full 30-epoch schedule and keeping the final epoch.
- **Efficiency**: `scripts/benchmark_models.py` on an otherwise idle machine
  (NVIDIA GeForce RTX 5060 Ti, AMD Ryzen 5 9600X with 6 physical cores):
  - GPU: median single-image latency and batch-256 throughput, PyTorch FP32
  - CPU server: median single-image latency, ONNX Runtime, 6 threads
  - Edge proxy: median single-image latency, ONNX Runtime, 1 thread

## Candidates

- The five off-the-shelf architectures of `reports/experiments/architecture_comparison.md`, each at
  the learning rate chosen by its per-architecture comparison.
- The custom slim ResNets on the Pareto front of the search (validation errors vs FLOPs) that are
  retrained with 3 seeds for the full schedule.

**Learning-rate check for finalists.** The architecture comparison suggested that some architectures
prefer a learning rate above 1e-3 when augmentation is on. Every candidate that would be selected for
a target under the rules below is therefore also trained with 3 seeds at 3e-3, if it was not already
at that rate. The rate with fewer mean errors is kept; if the two are within 4 errors, the original
rate is kept. The rules below are then applied again with the updated numbers.

## Search finalists

*Added after the screening search `slim_resnet_v1` (`reports/experiments/slim_resnet_v1.md`) and before
any confirmation run.*

The search screens each configuration with 15 epochs and a single seed. Screening errors are higher
than full-schedule errors, and a longer schedule may rank configurations differently, so more than the
single best trial is confirmed:

- **Finalists**: Pareto-optimal trials whose screening errors are at most **the best screening errors
  + 20**. The margin is wide on purpose: repeated configurations with similar hyperparameters varied by
  up to about 7 errors in the search, and the change of schedule adds uncertainty on top. In
  `slim_resnet_v1` the rule selects trials **15, 46, 10, 29 and 36** (74.7, 69.3, 61.0, 57.3 and 57.0
  screening errors; best 57.0, threshold 77.0). It excludes trials 42 and 11 (122.0 and 115.7 errors
  at 3.2 and 5.3 MFLOPs), twice the best errors, a gap a longer schedule cannot plausibly close.
- **Configs**: generated with `scripts/export_trial.py`: each trial's architecture, learning rate,
  weight decay and warmup, with the base experiment's full 30-epoch schedule.
- **Evidence**: 3 seeds each and `scripts/benchmark_models.py`, like every other candidate.

The search tuned learning rates for a 15-epoch schedule, which favours higher rates. The
learning-rate check for finalists described above therefore applies to them as to every other candidate.

## Eligibility

A candidate is eligible when its mean validation errors are at most **the best candidate's mean + 4**.
Four errors is the run-to-run noise observed with single seeds; within this margin, candidates are
treated as equally accurate.

## Selection per deployment target

Among eligible candidates:

| Target | Rule |
|---|---|
| Cloud GPU | Median single-image latency at most 5 ms; among those, the highest batch-256 throughput |
| CPU server | Lowest CPU-server latency |
| Edge | Lowest edge-proxy latency |

One model serves both GPU modes (single requests and batch review): the mode is a property of the
serving layer (dynamic batching), not of the model. The latency bound keeps the single-request mode
responsive; throughput decides the batch mode.

**Ties.** If two candidates are within 10% on a target's speed metric, prefer the model already
selected for another target, then the one with fewer parameters. Fewer distinct models mean less to
validate, monitor and retrain; a single model for all three targets is an acceptable outcome.

## After selection

1. Tune one decision threshold per class on the validation split, for each selected model.
2. Evaluate each selected model once on the test split, with those thresholds.
3. Report the test results as they are. A disappointing test result is reported, not fixed by
   changing the model, the thresholds or this protocol.
