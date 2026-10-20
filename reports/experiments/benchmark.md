Hardware: GPU NVIDIA GeForce RTX 5060 Ti; CPU AMD Ryzen 5 9600X 6-Core Processor (6 physical cores); torch 2.14.0+cu130, onnxruntime 1.30.0.

Single-image latency in ms, as median (p95). GPU: PyTorch FP32, input already on the GPU. CPU server: ONNX Runtime, 6 threads. Edge proxy: ONNX Runtime, 1 thread(s), only meaningful to rank models, not as an edge-device latency.

| Model | Params (M) | MFLOPs | ONNX (MB) | GPU latency | GPU batch 256 (img/s) | CPU server latency | Edge proxy latency |
|---|---|---|---|---|---|---|---|
| resnet18 | 11.17 | 3,047 | 44.7 | 1.43 (3.20) | 3,796 | 3.77 (4.20) | 10.54 (11.49) |
| mobilenet_v3_small | 1.53 | 94 | 6.3 | 1.84 (6.52) | 11,079 | 0.34 (0.39) | 0.64 (0.68) |
| mobilenet_v3_large | 4.21 | 355 | 17.1 | 2.29 (3.72) | 4,063 | 0.86 (0.98) | 1.86 (2.00) |
| efficientnet_b0 | 4.02 | 646 | 16.5 | 3.78 (5.34) | 2,357 | 1.57 (1.80) | 3.51 (3.93) |
| shufflenet_v2_x1_0 | 1.26 | 250 | 5.3 | 2.31 (3.16) | 7,336 | 0.82 (1.10) | 1.43 (1.64) |
| slim_resnet_w16_d2_std | 0.70 | 18 | 2.9 | 0.90 (8.07) | 180,685 | 0.07 (0.10) | 0.10 (0.14) |
| slim_resnet_w24_d1_std | 0.69 | 20 | 2.8 | 0.74 (0.95) | 212,927 | 0.06 (0.10) | 0.12 (0.16) |
| slim_resnet_w32_d1_std | 1.23 | 33 | 5.0 | 0.64 (1.06) | 162,971 | 0.08 (0.11) | 0.14 (0.18) |
| slim_resnet_w16_d2_small | 0.70 | 192 | 2.9 | 1.13 (1.71) | 24,672 | 0.28 (0.33) | 0.72 (0.77) |
| slim_resnet_w32_d1_small | 1.23 | 349 | 5.0 | 0.71 (5.18) | 18,166 | 0.45 (0.51) | 1.17 (1.32) |
