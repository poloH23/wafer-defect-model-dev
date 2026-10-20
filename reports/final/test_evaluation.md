# Test evaluation

The single evaluation on the test split (3,700 samples), following docs/model_selection.md. Thresholds are tuned per class on the validation split for each seed. Errors: samples with any label wrong. Mean ± standard deviation over seeds [0, 1, 2]; 95% intervals from 2,000 bootstrap resamples of test samples.

Code and protocol: commit '3a78a43e5fd0878526702e654204deab5d8cfa38'.

**Selected for all deployment targets: wafer_shufflenet_v2_x1_0 (learning_rate=0.003).** The other candidates are evaluated for transparency only: their results do not change the selection.

## All candidates

| Candidate | Val errors (0.5) | Test errors (0.5) | Test errors (tuned) | 95% interval | Exact match | Macro F1 | Macro AP |
|---|---|---|---|---|---|---|---|
| **wafer_shufflenet_v2_x1_0 (learning_rate=0.003)** (selected) | 43.7 ± 2.5 | 45.7 ± 2.1 | 46.0 ± 1.7 | 34.3 to 58.7 | 0.9876 ± 0.0005 | 0.9932 ± 0.0004 | 0.9995 ± 0.0000 |
| wafer_efficientnet_b0 | 42.0 ± 1.7 | 40.0 ± 1.0 | 42.7 ± 1.5 | 32.7 to 54.3 | 0.9885 ± 0.0004 | 0.9934 ± 0.0004 | 0.9997 ± 0.0001 |
| wafer_mobilenet_v3_large | 45.7 ± 2.3 | 43.7 ± 2.9 | 45.3 ± 4.0 | 34.3 to 57.3 | 0.9877 ± 0.0011 | 0.9963 ± 0.0030 | 0.9997 ± 0.0001 |
| wafer_augment | 43.3 ± 4.9 | 42.3 ± 2.1 | 48.7 ± 3.1 | 37.0 to 61.3 | 0.9868 ± 0.0008 | 0.9917 ± 0.0021 | 0.9993 ± 0.0003 |
| wafer_mobilenet_v3_small | 46.3 ± 0.6 | 47.3 ± 7.6 | 49.3 ± 8.3 | 38.3 to 61.7 | 0.9867 ± 0.0023 | 0.9962 ± 0.0023 | 0.9996 ± 0.0001 |
| wafer_slim_resnet_v1_t029 | 47.7 ± 2.1 | 49.3 ± 2.5 | 52.7 ± 1.5 | 40.7 to 66.0 | 0.9858 ± 0.0004 | 0.9931 ± 0.0004 | 0.9992 ± 0.0003 |
| wafer_slim_resnet_v1_t036 | 54.0 ± 9.0 | 56.3 ± 5.5 | 59.7 ± 7.1 | 46.7 to 73.3 | 0.9839 ± 0.0019 | 0.9926 ± 0.0003 | 0.9992 ± 0.0003 |
| wafer_slim_resnet_v1_t010 | 57.3 ± 5.9 | 58.0 ± 2.6 | 61.0 ± 2.0 | 48.3 to 74.3 | 0.9835 ± 0.0005 | 0.9909 ± 0.0025 | 0.9987 ± 0.0005 |
| wafer_slim_resnet_v1_t046 | 63.0 ± 10.6 | 69.7 ± 5.5 | 70.0 ± 6.1 | 56.3 to 85.0 | 0.9811 ± 0.0016 | 0.9922 ± 0.0009 | 0.9993 ± 0.0003 |
| wafer_slim_resnet_v1_t015 | 71.3 ± 10.0 | 71.7 ± 1.5 | 76.3 ± 2.5 | 63.3 to 91.7 | 0.9794 ± 0.0007 | 0.9898 ± 0.0028 | 0.9991 ± 0.0003 |

## Selected configuration against each other candidate

Seed-averaged test errors (tuned thresholds) of the selected configuration minus those of the other candidate, on identical resamples. Negative: the selected configuration makes fewer errors. An interval containing 0 means no detectable difference.

| Other candidate | Difference | 95% interval | Resamples where the selected one makes fewer errors |
|---|---|---|---|
| wafer_efficientnet_b0 | +3.3 | -3.3 to +9.7 | 15% |
| wafer_mobilenet_v3_large | +0.7 | -6.0 to +7.7 | 41% |
| wafer_augment | -2.7 | -9.3 to +4.3 | 77% |
| wafer_mobilenet_v3_small | -3.3 | -11.3 to +4.7 | 77% |
| wafer_slim_resnet_v1_t029 | -6.7 | -13.3 to +0.3 | 97% |
| wafer_slim_resnet_v1_t036 | -13.7 | -21.3 to -6.3 | 100% |
| wafer_slim_resnet_v1_t010 | -15.0 | -23.7 to -6.3 | 100% |
| wafer_slim_resnet_v1_t046 | -24.0 | -34.3 to -14.3 | 100% |
| wafer_slim_resnet_v1_t015 | -30.3 | -41.0 to -20.3 | 100% |

## Selected configuration per class (tuned thresholds, mean over seeds [0, 1, 2])

| Class | Test support | Precision | Recall | F1 | AP | Thresholds (per seed) |
|---|---|---|---|---|---|---|
| Center | 1200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.50, 0.50, 0.50 |
| Donut | 1200 | 0.9997 | 1.0000 | 0.9999 | 1.0000 | 0.50, 0.50, 0.50 |
| Edge_Loc | 1200 | 0.9986 | 0.9894 | 0.9940 | 0.9981 | 0.80, 0.75, 0.40 |
| Edge_Ring | 1200 | 0.9947 | 0.9992 | 0.9970 | 0.9990 | 0.05, 0.30, 0.50 |
| Loc | 1800 | 0.9989 | 0.9943 | 0.9966 | 0.9990 | 0.40, 0.50, 0.50 |
| Near_Full | 15 | 0.9375 | 1.0000 | 0.9677 | 1.0000 | 0.50, 0.50, 0.50 |
| Scratch | 1800 | 0.9974 | 0.9920 | 0.9947 | 0.9998 | 0.55, 0.40, 0.30 |
| Random | 85 | 1.0000 | 0.9922 | 0.9961 | 1.0000 | 0.55, 0.45, 0.50 |

**Deliverable**: seed 0 of the selected configuration (MLflow run 176643c399f14d199631a45c01f3f427), with its tuned thresholds.
