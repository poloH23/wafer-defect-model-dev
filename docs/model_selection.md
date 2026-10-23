# Model Selection Protocol

> Languadge : *[繁體中文](zh-TW/model_selection.md)*

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

## Outcome of the selection

*Added after the confirmation runs, before any evaluation on the test split.*

Mean validation errors over 3 seeds, computed without rounding (the comparison table shows one
decimal, too coarse for the margins below):

| Candidate | Validation errors | Eligible |
|---|---|---|
| EfficientNet-B0 | 42.0667 | yes (best) |
| ShuffleNetV2 1.0x (learning rate 3e-3) | 43.0667 | yes |
| MobileNetV3-Large | 43.9333 | yes |
| ResNet-18 | 44.0667 | yes |
| MobileNetV3-Small | 46.2000 | no |
| Slim ResNets (trials 29, 36, 10, 46, 15) | 48.7333 to 71.6667 | no |

The eligibility threshold is 42.0667 + 4 = 46.0667. Among the four eligible candidates:

- **Cloud GPU**: all have a median latency under 5 ms; ShuffleNetV2 has the highest batch-256
  throughput (7,336 images/s; next MobileNetV3-Large, 4,063).
- **CPU server**: ShuffleNetV2 (0.82 ms) and MobileNetV3-Large (0.86 ms) are within 10%, a tie
  resolved in favor of ShuffleNetV2, already selected for the other targets. In the first benchmark
  run ShuffleNetV2 was faster outright (0.74 vs 0.85 ms).
- **Edge**: ShuffleNetV2 (1.43 ms; next MobileNetV3-Large, 1.86 ms).
ShuffleNetV2 already uses a learning rate of 3e-3, so the learning-rate check does not apply.
**ShuffleNetV2 1.0x with a learning rate of 3e-3 is selected for all three deployment targets.**

**Sensitivity.** The outcome rests on a margin of 0.1333 errors: MobileNetV3-Small misses
eligibility by that much, and if eligible it would have been selected for all three targets
(11,079 images/s, 0.34 ms and 0.64 ms). Before the exact values were computed, both outcomes and
their consequences were written down, so the margin could not influence the decision.

**A flaw of this protocol.** The learning-rate check only covers candidates that would be selected,
so a candidate excluded because its learning rate did not suit it is never checked. MobileNetV3-Small
is such a case: its single-seed run at 3e-3 had 43.6 errors, 2.6 fewer than its mean at 1e-3, and a
check covering candidates near the threshold might have made it eligible. The flaw is reported here
rather than fixed, since changing the rule after seeing whom the change favors is what this protocol
exists to prevent.

**Unequal tuning.** The slim ResNets were tuned more than the off-the-shelf architectures (50 screening
trials over learning rate, weight decay and warmup, against three learning rates). This cannot have
helped them here, as none became eligible, but it should be kept in mind in any comparison of their
accuracy.

## Test evaluation

*Added before the test split was used for anything.* The test split is evaluated once, with
`scripts/evaluate_test.py` and `conf/evaluation/final_test.yaml`; the script refuses to overwrite an
existing report.

1. **Models.** All 3 seeds of the selected configuration, as trained on the training split. They
   are not retrained on training plus validation data: the validation split must stay unseen by
   training to tune the decision thresholds.
2. **Thresholds.** For each model (each seed) and each class, the threshold that maximizes that
   class's F1 on the validation split, among 0.05 to 0.95 in steps of 0.05. Ties go to the value
   closest to 0.5, the untuned default.
3. **Primary result.** Test errors (samples with any label wrong) at the tuned thresholds, as mean
   and standard deviation over the 3 seeds. Also reported: errors at the 0.5 default (the effect of
   tuning), exact match, macro F1, macro average precision, and per-class precision, recall and F1.
4. **Two kinds of uncertainty.** Training randomness, as the standard deviation over seeds. The
   finite test set, as a 95% bootstrap confidence interval of the seed-averaged errors (2,000
   resamples of test samples, fixed seed, percentile method).
5. **Every other candidate, for transparency only.** The same procedure is applied to the other nine
   candidates, each at its pre-declared learning rate, and reported with the paired difference
   between the selected configuration and each of them: the difference in seed-averaged errors on
   identical bootstrap resamples of the test set, with its 95% confidence interval. These results show
   whether the validation-based selection holds on unseen data; **they do not change the selection**.
6. **Deliverable.** The seed-0 model of the selected configuration, with its tuned thresholds. It is
   fixed in advance rather than picked as the best seed, which would be one more selection on noise.
7. **Reporting.** Results are committed as they come out. A disappointing result is reported, not
   fixed by changing the model, the thresholds or this protocol.
