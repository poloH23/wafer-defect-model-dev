# Methodology

How the models were developed and evaluated, and why each step was designed the way it was.
Detailed results are in `reports/experiments/`; the rules that choose the final models are fixed in
[model_selection.md](model_selection.md), committed before the search and the test evaluation.

The project moved through one question at a time. Each answer shaped the next experiment:

1. Is the evaluation itself trustworthy? → a protocol whose run-to-run noise is measured and small
2. Where do the remaining errors come from? → mostly ambiguous labels, not model weakness
3. Does capacity matter, then? → no: architectures 32x apart in compute reach the same accuracy
4. How does compute translate into speed? → it depends on the hardware, so it was measured
5. How small can a model get? → a search that maps accuracy against compute

## 1. Data and evaluation protocol

**Data.** MixedWM38: 38,015 wafer maps of 52x52 dies (0 = outside the wafer, 1 = pass, 2 = fail),
each with any combination of 8 defect labels. Two data-quality problems were found and handled
before any modeling (see *Data Quality Findings* in the README): 214 pixels with an undocumented value
3, imputed in memory from their neighbors, and 1,023 groups of identical images. Split `v1` left 157
validation and 174 test images with an identical copy in training, which inflated every score; split
`v2` removes the 1,024 duplicates before splitting.

**Split.** 80/10/10, stratified by the exact label combination rather than by single labels, so rare
combinations appear in every split. Each split records a digest of its indices and the hash of the
data file; every training run checks both, so a result can always be traced to the exact data.

**Validation only.** Every decision in this document uses the validation split. The test split is
evaluated once, at the end, for the selected models only.

**Accuracy metric: misclassified samples.** A validation sample counts as an error when any of its
8 labels is wrong at a 0.5 threshold. Macro average precision, the first choice, saturated around
0.998: every model scored the same, so it could no longer rank them.

**Training protocol.** Every run trains for the full 30-epoch cosine schedule and keeps the final
epoch. Keeping the best epoch instead picks the luckiest of 30 noisy evaluations of the same
validation set, which biases scores upward; early stopping on top of it cut the learning-rate
annealing short. The change reduced the seed-to-seed standard deviation of the baseline's macro AP
from 0.0009 to 0.00015.

**Comparisons.** Each run is summarized by the mean of its last 5 epochs, each configuration by the
mean and standard deviation over 3 seeds. A difference smaller than these standard deviations is not
treated as evidence; with a single seed, about 4 errors is the observed noise.

## 2. Where the remaining errors come from

An error analysis of the baseline found that the most frequent confusion is between Edge_Loc (a
defect on part of the wafer edge) and Edge_Ring (a defect along the whole edge). To test whether the
model or the labels are at fault, a rule-based **edge coverage** was computed for every sample: the
share of the circumference whose outermost band consists mostly of failed dies.

- Coverage separates the two labels well overall: ROC AUC 0.987 to 0.996 for sector thresholds from
  0.4 to 0.6, so the result does not depend on that parameter.
- All 14 validation samples where the baseline predicted Edge_Ring instead of Edge_Loc lie in the top
  4% of Edge_Loc coverage, inside the typical range of Edge_Ring. Three of them cover the entire edge.
- Prediction probabilities show the model is confident in these decisions: it follows the class
  definitions, and the labels contradict them in that region.

About a third of the remaining errors are of this kind. This is the evidence behind the project's
central decision: accuracy is limited by label quality, so beyond a point, effort goes into
efficiency rather than into more accuracy.

## 3. Ablation: augmentation

Random dihedral transforms (90-degree rotations and flips) are exact on discrete pixel values and do
not change what a wafer-map defect is. With 3 paired seeds (same initialization and data order, the
only difference being augmentation):

| | Validation errors | Validation loss at the end |
|---|---|---|
| No augmentation | 52.8 ± 2.7 | 0.013 to 0.014 |
| Augmentation | 44.1 ± 4.2 | 0.0068 to 0.0072 |

Beyond fewer errors, the halved validation loss means better-calibrated probabilities: without
augmentation the model memorized the training set and was confidently wrong. Augmentation was kept.

## 4. Architecture comparison

Five torchvision architectures were trained from scratch with the same protocol. All of them were
designed for 224x224 photos and downsample 32x in five stride-2 steps; each was adapted the same way,
by removing the first two steps, so a 52x52 map still reaches the classifier as a 7x7 feature map.

**Learning rate per architecture.** A single shared learning rate can favor some architectures. Each
architecture was therefore run with seed 0 at 3e-4, 1e-3 and 3e-3, under a rule fixed beforehand:
the fewest errors wins; a best value at the edge of the grid is extended one step further; if the top
two are within 4 errors, 1e-3 is kept. ShuffleNetV2 was extended to 1e-2 and kept 3e-3. With the
shared 1e-3 it would have been penalized by 6 errors, a difference due to the learning rate, not the
architecture.

| Architecture | Params | MFLOPs | Validation errors (3 seeds) |
|---|---|---|---|
| EfficientNet-B0 | 4.02M | 646 | 42.1 ± 2.0 |
| ShuffleNetV2 1.0x (lr 3e-3) | 1.26M | 250 | 43.1 ± 2.1 |
| MobileNetV3-Large | 4.21M | 355 | 43.9 ± 1.0 |
| ResNet-18 | 11.17M | 3,047 | 44.1 ± 4.2 |
| MobileNetV3-Small | 1.53M | 94 | 46.2 ± 0.2 |

Accuracy is the same within noise across a 32x range of compute, as expected if capacity is not the
bottleneck. The grid also suggested that 3e-3 suits several architectures better than 1e-3 once
augmentation is on. The rule was not changed after the fact; the observation was carried forward
instead, into the search's learning-rate range and a learning-rate check for finalists.

## 5. Efficiency measurement

`scripts/benchmark_models.py` measures each architecture for the three deployment targets on an RTX
5060 Ti and a Ryzen 5 9600X: GPU single-image latency and batch-256 throughput (PyTorch FP32, timed
with CUDA events), CPU-server latency (ONNX Runtime, 6 threads) and an edge proxy (ONNX Runtime,
1 thread, which ranks models but is not an edge-device latency). Each ONNX export is checked against
PyTorch outputs before being timed.

| Architecture | GPU latency (ms) | GPU throughput (img/s) | CPU server (ms) | Edge proxy (ms) |
|---|---|---|---|---|
| ResNet-18 | 1.46 | 3,800 | 3.79 | 10.39 |
| MobileNetV3-Small | 2.24 | 11,070 | 0.38 | 0.65 |
| MobileNetV3-Large | 2.67 | 4,050 | 0.85 | 1.86 |
| EfficientNet-B0 | 4.03 | 2,407 | 1.91 | 3.59 |
| ShuffleNetV2 1.0x | 2.08 | 7,332 | 0.74 | 1.42 |

- **On CPU, FLOPs predict latency within an architecture family, but not exactly across families.**
  Among the five off-the-shelf architectures, FLOPs rank the models exactly as measured latency does.
  Benchmarking the slim ResNets later showed the limit: with 6 threads, ResNet-style dense
  convolutions complete about 690 to 810 MFLOPs per millisecond, the mobile architectures only 280
  to 410, so a slim ResNet with twice the FLOPs of MobileNetV3-Small is still faster on the CPU
  server (0.28 vs 0.34 ms).
- **On GPU, single-image latency follows the number of layers, not FLOPs**: ResNet-18, with the most
  FLOPs but few layers, is the fastest; EfficientNet-B0, with the most layers, the slowest. With a
  batch of 256 the GPU is busy enough for compute to matter again, but depthwise convolutions are
  limited by memory bandwidth, so speed grows far less than FLOPs shrink.

This supported FLOPs as the search objective for the CPU and edge targets, as long as the search stays
within one architecture family, and showed that GPU speed, like any comparison across families, has
to be measured directly. It also shaped the GPU target: one model serves both single requests and
batch review, since the mode belongs to the serving layer (dynamic batching), and fewer models mean
less to validate, monitor and retrain.

## 6. Multi-objective search over slim ResNets

### Purpose

To map the trade-off between accuracy and compute for an architecture that GPUs run efficiently, and
supply candidates for the three deployment targets. The search informs a decision; it is not meant to
find a single best model.

### Hypotheses

1. **Capacity is not the bottleneck**, so a ResNet can shrink a long way before accuracy drops. The
   search should find where it starts dropping.
2. **A slim ResNet serves both CPU and GPU**: few dense layers, which GPUs handle well, with FLOPs
   down at the level of mobile architectures. Verified by benchmarking, not by the search.
3. **Augmentation favors higher learning rates**, from the architecture comparison.
4. **Warmup tames the unstable first epochs** seen in every run so far.

### Design

| Decision | Choice | Reason |
|---|---|---|
| Objectives | Validation errors and FLOPs, both minimized | The trade-off is decided later, per target; a weighted sum would fix it in advance |
| Accuracy measure | Errors averaged over the last 3 epochs | Same metric as the final selection; macro AP cannot rank models |
| Efficiency measure | FLOPs | Exact and hardware-independent; its ranking matched measured CPU latency |
| Architecture space | Slim ResNet: width 8 to 64, 1 or 2 blocks per stage, small or standard stem | Width 8 probes where accuracy drops; width 64 with 2 blocks is exactly ResNet-18 |
| Training space | Learning rate 3e-4 to 1e-2, weight decay 1e-5 to 1e-3 (log scale), warmup 0 to 3 epochs | Learning-rate range widened upward by the architecture comparison |
| Dimensionality | Same depth in all four stages | Keeps the space in proportion to 50 trials |
| Fidelity | 15 epochs, 1 seed, 50 trials; finalists confirmed with 30 epochs and 3 seeds | Optuna cannot prune multi-objective trials, so each trial must be cheap |
| Sampler | Optuna TPE, seeded; the study is stored and resumable | The first 10 trials explore at random, the rest concentrate on promising regions |
| Controls | Every other setting from the base experiment; validation split only | Differences between trials come from the searched parameters alone |

Every trial is a complete MLflow run that logs its effective configuration and a checkpoint, so any
point of the trade-off curve can be traced, analyzed and retrained.

### Screening results

Single-seed, 15-epoch results; not conclusions until confirmed with the full protocol.

- **Accuracy is flat from about 33 MFLOPs upward**: from 33 to 349 MFLOPs, ten times the compute,
  the best errors fall only from 61.0 to 57.0. Below about 20 MFLOPs they rise steeply, doubling by
  3 MFLOPs. Hypothesis 1 holds down to roughly a third of MobileNetV3-Small's compute, and the search
  located its limit.
- **At low compute, the standard stem wins**: downsampling the input early and spending the compute
  on more channels beat keeping the full resolution with fewer channels, against the assumption that
  motivated the small-input stem. At the most accurate end, the small-input stem still wins.
- **Learning rates below 1e-3 always did badly**, and the best trials sat near the upper bound.

### What the search taught about designing searches

- **Check the boundaries.** The best learning rates sat near the upper bound of 1e-2 and the best
  weight decays near the lower bound of 1e-5: the optimum may lie outside the range.
- **Low fidelity biases some hyperparameters.** A 15-epoch schedule favors high learning rates, so
  part of that preference may be an artifact of the screening itself. Hyperparameters tied to the
  schedule are the most exposed, which is why finalists are confirmed with the full schedule.
- **A cheap objective attracts the sampler.** FLOPs is exact and easy to reduce, so 28 of 50 trials
  landed below 40 MFLOPs and none above 800: the expensive end of the curve stayed unexplored.
- **A proxy is valid within a scope.** FLOPs ranked CPU latency perfectly within each architecture
  family and imperfectly across families, where dense and depthwise convolutions use the hardware
  differently. The search compared slim ResNets only, the scope in which its proxy holds; comparisons
  across families use measured latency.
- **Enqueue reference points.** Evaluating a known configuration such as the ResNet-18 equivalent at
  the start (`study.enqueue_trial`) would have given an anchor at the same fidelity, and measured the
  gap between 15 and 30 epochs.
- **Repeated configurations measure noise for free.** The sampler drew one architecture seven times;
  with similar hyperparameters, its errors stayed within 1.7.
- **Searches get interrupted.** Optuna does not store its sampler's random state, so a resumed search
  seeded as before replayed its first suggestions. The sampler is now reseeded with the number of
  trials already stored.

## 7. Confirmation and selection

The search finalists are retrained with 3 seeds and the full schedule, benchmarked, and compared with
the off-the-shelf architectures. The rules that turn this evidence into one model per deployment
target are in [model_selection.md](model_selection.md); they were committed before the search, the
confirmation runs and the test evaluation. Each selected model then gets per-class decision thresholds
tuned on the validation split, and is evaluated once on the test split.
