# Methodology

How the models were developed and evaluated, and why each step was designed the way it was.
Detailed results are in `reports/experiments/`; the rules that choose the final models are fixed in
[model_selection.md](model_selection.md), committed before the search and the test evaluation.

The project moved through one question at a time. Each answer shaped the next experiment:

1. Is the evaluation itself trustworthy? → a protocol whose run-to-run noise is measured and small
2. Where do the remaining errors come from? → mostly ambiguous labels, not model weakness
3. Does capacity matter, then? → no: architectures 32x apart in compute reach the same accuracy
4. How does compute translate into speed? → it depends on the hardware, so it was measured
5. How small can a model get? → a search mapped the trade-off; its most promising finding did not
   survive confirmation
6. Does the choice hold on unseen data? → yes, and the test evaluation exposed two flaws of the
   selection protocol
7. What does the selected model look at? → compact defects where they are; for Edge_Loc, the places
   where the edge stops failing, which also explains the label confusion found in step 2

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

Single-seed, 15-epoch results, recorded here as they were seen. The first finding did not survive the
full-protocol confirmation (section 7).

- **In screening, accuracy looked flat from about 33 MFLOPs upward**: from 33 to 349 MFLOPs, ten times
  the compute, the best errors fell only from 61.0 to 57.0, and rose steeply below about 20 MFLOPs.
  It suggested that hypothesis 1 held down to roughly a third of MobileNetV3-Small's compute.
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
- **Screening can mislead in level and in ranking.** Confirmed with 3 seeds and the full schedule, the
  apparent plateau became a 10-error gap between 33 and 192 MFLOPs, and the two best screening trials
  swapped places. Single-seed, low-fidelity results locate candidates; they do not support conclusions.
- **Repeated configurations measure noise for free.** The sampler drew one architecture seven times;
  with similar hyperparameters, its errors stayed within 1.7.
- **Searches get interrupted.** Optuna does not store its sampler's random state, so a resumed search
  seeded as before replayed its first suggestions. The sampler is now reseeded with the number of
  trials already stored.

## 7. Confirmation and selection

The search finalists were the Pareto-optimal trials within 20 screening errors of the best, a rule
written down before they were retrained: five slim ResNets, retrained with 3 seeds and the full
30-epoch schedule. Mean validation errors, computed without rounding:

| Candidate | MFLOPs | Validation errors |
|---|---|---|
| EfficientNet-B0 | 646 | 42.07 |
| ShuffleNetV2 1.0x (learning rate 3e-3) | 250 | 43.07 |
| MobileNetV3-Large | 355 | 43.93 |
| ResNet-18 | 3,047 | 44.07 |
| MobileNetV3-Small | 94 | 46.20 |
| Slim ResNet, width 16, 2 blocks, small stem | 192 | 48.73 |
| Slim ResNet, width 32, 1 block, small stem | 349 | 53.13 |
| Slim ResNet, width 32, 1 block, standard stem | 33 | 58.67 |
| Slim ResNet, width 24, 1 block, standard stem | 20 | 64.07 |
| Slim ResNet, width 16, 2 blocks, standard stem | 18 | 71.67 |

**What this says about the hypotheses.** None of the slim ResNets is as accurate as the off-the-shelf
architectures, even at similar compute: at 192 to 349 MFLOPs they make 5 to 10 more errors than
ShuffleNetV2 and MobileNetV3-Large at 250 to 355. Hypothesis 1 holds for the off-the-shelf
architectures, whose designs (depthwise convolutions, and in some squeeze-and-excitation blocks)
appear to get more accuracy out of each FLOP, but not for a plain ResNet shrunk down. Hypothesis 2 holds for speed, the slim ResNets being the fastest models on every target,
but at an accuracy cost that excludes them. A negative result, and the reason the confirmation stage
exists: selecting from the screening alone would have chosen a model that the full protocol shows to
be less accurate.

**Selection.** The rules of [model_selection.md](model_selection.md) were then applied: a candidate is
eligible within 4 errors of the best (42.07 + 4 = 46.07), and each target takes the fastest eligible
one. ShuffleNetV2 has the highest GPU throughput and the lowest CPU and edge latency among the four
eligible candidates, so **one model, ShuffleNetV2 1.0x, serves all three targets**. MobileNetV3-Small,
faster on every target, missed eligibility by 0.13 errors. Since the comparison table rounds to one
decimal, exact values were computed, after both possible outcomes and their consequences had been
written down, so that the margin could not influence the decision.

## 8. What the test evaluation showed

Each candidate was evaluated once on the test split, with per-class thresholds tuned on the
validation split; the full report is `reports/final/test_evaluation.md`. The selected model makes
46.0 ± 1.7 errors out of 3,700 test samples (95% interval 34.3 to 58.7), an exact match of 98.76%.

- **Validation conclusions generalize.** Test and validation errors differ by 1 to 2 for every
  off-the-shelf candidate. Many decisions were taken on the validation split; none overfitted it.
- **The finite test set dominates uncertainty.** Across seeds, the selected model's errors vary by a
  standard deviation of 1.7; across bootstrap resamples of the test set, its 95% interval spans about
  ±12. Comparing two models on the same resamples cancels the samples both get wrong and narrows the
  interval of their difference to about ±7.
- **The selection logic is confirmed.** No off-the-shelf architecture differs detectably from the
  selected model, while four of the five slim ResNets are significantly less accurate.
- **Threshold tuning did not help.** Tuned thresholds increased test errors for all ten candidates, by
  2.8 on average. They maximized per-class F1 while the primary metric counts samples with any wrong
  label, and with 40 to 70 validation errors per model, each threshold followed a handful of samples:
  for the selected model, the Edge_Ring threshold ranged from 0.05 to 0.50 across seeds.

### Lessons for the next selection protocol

The test evaluation exposed two flaws. They are reported, not corrected, since changing the rules
after seeing the results would defeat the protocol; the next protocol should avoid them.

- **Decide equivalence with paired comparisons, not a fixed margin.** The 4-error margin reflected
  training noise only, while the validation set's own sampling noise is about ±12 errors. Widening the
  margin to 12 would not have fixed it: it would also have admitted two slim ResNets, one of them the
  fastest eligible model on GPU and CPU, and both less accurate on the test set (one significantly, the
  other nearly so). The principled criterion is the one applied to the test set: a paired bootstrap
  comparison of each candidate with the best one on the validation split, accepting a candidate when
  the upper bound of the interval of its extra errors stays below a tolerance fixed in advance (a
  non-inferiority test).
- **Give every candidate the same tuning.** The learning-rate check covered only candidates that would
  be selected, so a candidate excluded by an unsuitable learning rate was never rechecked.
- **Align threshold tuning with the primary metric, and require it to beat noise.** Tune thresholds to
  minimize the samples with any wrong label, and keep 0.5 unless the tuned thresholds improve on it by
  more than validation noise.

## 9. What the model looks at

`scripts/explain_gradcam.py` computes Grad-CAM heatmaps of the delivered model (ShuffleNetV2, seed 0)
on the validation split, from its last convolutional block: 7x7, upsampled to the 52x52 map. The
analysis is descriptive and changes nothing in the model. Full report: `reports/final/gradcam.md`.

### Design: heatmaps are evidence only with a control

A heatmap that looks plausible proves little: some saliency methods produce nearly the same maps for
a trained network and for one with random weights, because they reflect the input's structure rather
than what the network learned (Adebayo et al., 2018, *Sanity Checks for Saliency Maps*). Every
measurement is therefore repeated on the same architecture with random weights, and examples are
drawn at random with a fixed seed rather than picked for how they look.

- **Localization.** Center, Donut and the edge classes lie in a known zone by definition. The wafer is
  split into three concentric zones (center up to 0.3 of the radius, ring to 0.7, edge beyond), and two
  measures are taken on every correctly classified single-defect sample: how often the heatmap's peak
  falls in the class's zone (the pointing game, whose chance level is the zone's share of the wafer),
  and the zone's share of the heatmap mass relative to its share of the area.
- **Direction.** For Edge_Loc, a defect on part of the edge: the angle between where the heatmap
  points and where the edge's failed dies lie, seen from the center.

### Calibrating the instrument before using it

The measures were first run on synthetic wafers, where the defect positions are known. The first
version used a 5-pixel edge band and reported no focus for Edge_Loc heatmaps that visibly sat on the
defect: at 7x7, a heatmap resolves direction well but radius only to about one cell, and its peak sits
one cell inward of an edge defect. The edge zone was widened to one cell and the direction measure
added, both before any real data was measured. Adjusting a measure until a result looks right would
distort the evidence; fixing it against a known answer is calibration.

### Results

- **The randomization check passes for every class.** Trained and random-weight heatmaps are nearly
  uncorrelated (rank correlation 0.01 to 0.16): the maps reflect what the model learned.
- **Center is localized precisely.** The heatmap's peak is in the center zone for all 100 samples
  (random weights: 8%, chance: 9%), with 3.49 times the mass that area alone would give (random: 0.92).
- **Compact defects look right in the examples.** Scratch heatmaps follow the scratch line, Loc
  heatmaps cover the local cluster, Near_Full heatmaps most of the wafer.
- **Global patterns give diffuse heatmaps.** Edge_Ring and Random heatmaps spread over the wafer; a
  7x7 Grad-CAM localizes compact evidence, not a pattern spread along the whole edge.
- **One zone was wrong.** Only 4% of Donut peaks fall in the ring zone, below chance, although Donut
  heatmaps carry more mass in the middle of the wafer than area alone would give (1.49, random: 1.03).
  The maps suggest why: the donuts visible in the figures lie at about 0.2 to 0.35 of the radius,
  mostly inside the center zone. The zone came from an assumed geometry rather than from the data; it was not redefined
  after seeing the heatmaps, which would have fitted the measure to the result.
- **The random-weight control has a bias of its own.** Its peaks fall in the edge zone for 12% of
  Edge_Loc samples, far below the 51% the area would give, likely because zero padding weakens the
  outermost cells.
  For edge classes, the control is a better baseline than the nominal chance level.

### Why Edge_Loc heatmaps pointed the wrong way

The direction measure contradicted its calibration: on synthetic wafers, Edge_Loc heatmaps pointed at
the defect (median angle 12 degrees, all four samples within 45); on the real model, the median angle was 162
degrees and 25% fell within 45, the chance level. A diagnostic was designed to tell whether the
measure or the model was responsible.

**Method.** The edge is split into the 36 sectors of the edge coverage, each failed (most of its dies
failed) or intact. A sector next to one of the other kind is a boundary, where failed and intact parts
of the edge meet. For each kind of sector, the heatmap's share of mass is divided by the kind's share
of the sectors, so 1 means no preference. The measure does not average directions, so it holds for
wide defects and for heatmaps with several peaks. On synthetic wafers it found what was known: a
ratio of 3.39 for failed sectors, the favorite of all four samples.

**Predictions, written down before the real model was measured.** Failed sectors favored: the
heatmaps point at the defect and only the direction measure failed. Boundaries favored: they mark the
ends of the failed arc. Intact sectors favored: the model checks where the edge did not fail. No kind
clearly above the random-weight control: Grad-CAM cannot localize along the edge of real wafers.

**Result**, on 100 Edge_Loc samples whose failed arcs cover a median 56% of the edge:

| Model | Failed | Boundary | Intact | Favored most: failed / boundary / intact |
|---|---|---|---|---|
| Trained | 0.34 | 2.00 | 1.78 | 3% / 66% / 31% |
| Random weights | 1.06 | 1.03 | 1.04 | 37% / 28% / 35% |

The control shows no preference, so the pattern is learned. The heatmaps avoid the failed arc and
concentrate where it ends and on the edge that did not fail. In the profile figure, a wafer whose
edge fails all around except one gap has its heatmap peak in the gap; one with a wider gap, at both
of its ends.

**Interpretation.** What separates Edge_Loc from Edge_Ring is whether the edge fails all around. With
arcs covering half the edge, the decisive evidence for a *partial* edge defect is where the failure
stops, and that is what the model learned, a reason an expert would recognize. It accounts for every
earlier observation:

- **The 162-degree angle**: the mass sits at the ends of the arc and on the intact side, opposite the
  failed arc's mean direction.
- **The calibration's different answer**: the synthetic Edge_Loc arcs were narrow (70 degrees, a fifth
  of the edge), so the presence of a failed arc was the best evidence there. What a model looks at depends on what
  separates the classes in its data; the calibration showed the instrument can detect a preference for
  the defect, and on the real model it detected a different one.
- **The label confusion of section 2**: three of the 9 Edge_Loc samples predicted as Edge_Ring have an
  edge coverage of 1.00, an edge that fails all around and offers no place where the failure stops. Two
  of them have an empty Edge_Loc heatmap; the third keeps a small spot, perhaps where a sector is only
  partly failed. This is consistent with the mechanism, but the confused samples were not measured with
  the diagnostic, so it is supporting evidence, not proof.

### Lessons

- **Calibrate an instrument on known answers before using it**, and do not expect it to return the
  same answer afterwards: its job is to detect what is there.
- **List every possible outcome in advance, including "no conclusion".** Reading the result then means
  matching a row of a table, not building an explanation after the fact.
- **Derive geometric priors from the data**, such as each class's radial distribution of failed dies,
  rather than from an assumed shape.
- **Compare with an empirical control, not only a nominal chance level**: the random-weight model
  revealed a border bias that the area-based chance level ignores.
