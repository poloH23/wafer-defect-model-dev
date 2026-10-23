# Model card: wafer defect classifier

## Model

- **Architecture**: shufflenet_v2_x1_0, trained from scratch on MixedWM38 with rotation and flip augmentation.
- **Delivered model**: seed 0 of the configuration selected on the validation split for all three
  deployment targets (cloud GPU, CPU server, edge), following docs/model_selection.md.
- **Format**: a single ONNX file; the input encoding and the sigmoid are part of the graph, the thresholds and class
  names are in its metadata. Reference inference with NumPy and ONNX Runtime only: src/inference/classifier.py.

## Intended use

Multi-label classification of 52x52 wafer maps into 8 defect types, any combination of
which may be present. Trained and evaluated on a partly synthetic public dataset, not on production data: it supports
the analysis of wafer maps, but its decisions should be validated on the target production line before being relied on.

## Input and output

- **Input** 'wafer_map': uint8, shape (N, 52, 52); one category id per die, 0 outside the wafer, 1 pass,
  2 fail. Values of 3 or more are handled as in training (neighbor_mode).
- **Output** 'probabilities': float32, shape (N, 8), one probability per class, in the order below.
- **Decision**: a class is present when its probability reaches its threshold, tuned on the validation split.

| Class | Threshold |
|---|---|
| Center | 0.50 |
| Donut | 0.50 |
| Edge_Loc | 0.80 |
| Edge_Ring | 0.05 |
| Loc | 0.40 |
| Near_Full | 0.50 |
| Scratch | 0.55 |
| Random | 0.55 |

## Performance

Evaluated once on the held-out test split (3,700 maps), with per-class thresholds tuned on the validation split.
Errors are maps with any label wrong.

| | Errors | Exact match | Macro F1 | Macro average precision |
|---|---|---|---|---|
| Configuration, 3 seeds | 46.0 ± 1.7 (95% interval 34.3 to 58.7) | 98.76% ± 0.05% | 0.9932 ± 0.0004 | 0.9995 ± 0.0000 |
| Delivered model, seed 0 | 47 | 98.73% | 0.9929 | 0.9995 |

Per class, delivered model (seed 0):

| Class | Test maps | Precision | Recall | F1 |
|---|---|---|---|---|
| Center | 1200 | 1.0000 | 1.0000 | 1.0000 |
| Donut | 1200 | 0.9992 | 1.0000 | 0.9996 |
| Edge_Loc | 1200 | 0.9983 | 0.9892 | 0.9937 |
| Edge_Ring | 1200 | 0.9942 | 0.9992 | 0.9967 |
| Loc | 1800 | 0.9989 | 0.9944 | 0.9967 |
| Near_Full | 15 | 0.9375 | 1.0000 | 0.9677 |
| Scratch | 1800 | 0.9978 | 0.9917 | 0.9947 |
| Random | 85 | 1.0000 | 0.9882 | 0.9941 |

## Efficiency

1.26M parameters, 250 MFLOPs per map, 5.3 MB as ONNX. Median single-map latency (reports/experiments/benchmark.md):

- GPU (PyTorch FP32): 2.31 ms; 7,336 maps/s in batches
- CPU server (ONNX Runtime, multiple threads): 0.82 ms
- Edge proxy (ONNX Runtime, 1 desktop thread; ranks models, not a device latency): 1.43 ms

## Export verification

On the 3,700 maps of the validation split, raw maps through the ONNX file and the reference
inference against the PyTorch checkpoint on the training pipeline's inputs:

- Largest probability difference: 1.2e-06
- Maps whose decisions differ: 0
- Largest difference between batched and one-by-one inference: 0.0e+00

## Limitations

- Partly synthetic training data with an artificial class balance: precision on a production line will differ.
- Ambiguous labels between Edge_Loc and Edge_Ring: errors concentrate where the edge fails almost all around.
- Rare classes (Near_Full, Random) have few test maps, so their per-class figures are imprecise.
- See the Limitations section of the README for the full list, including those of the model selection.

## Provenance

- MLflow run: 176643c399f14d199631a45c01f3f427
- Training code commit: f61496b75d3edb93c199e1572b5d162b5d2989df; export code commit: 2c41554677412307e17f4c265b7c490f43133222
- Data split digest: e3b2c7a0e916baa268a14e91ee26aac53f54fd40eb4bd805925d87f4a6212706
- ONNX file SHA-256: bc39e046c7d687fc70f5bc89d4778e16ba8478ffd11f19a4a690b0eddfea924d
