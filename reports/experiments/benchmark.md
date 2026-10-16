Hardware: GPU NVIDIA GeForce RTX 5060 Ti; CPU AMD Ryzen 5 9600X 6-Core Processor (6 physical cores); torch 2.14.0+cu130, onnxruntime 1.30.0.

Single-image latency in ms, as median (p95). GPU: PyTorch FP32, input already on the GPU. CPU server: ONNX Runtime, 6 threads. Edge proxy: ONNX Runtime, 1 thread(s), only meaningful to rank models, not as an edge-device latency.

| Model | Params (M) | MFLOPs | ONNX (MB) | GPU latency | GPU batch 256 (img/s) | CPU server latency | Edge proxy latency |
|---|---|---|---|---|---|---|---|
| resnet18 | 11.17 | 3,047 | 44.7 | 1.55 (5.39) | 3,805 | 3.76 (4.21) | 10.42 (11.05) |
| mobilenet_v3_small | 1.53 | 94 | 6.3 | 2.25 (3.73) | 10,886 | 0.37 (0.50) | 0.63 (0.70) |
| mobilenet_v3_large | 4.21 | 355 | 17.1 | 2.08 (3.46) | 4,057 | 0.83 (0.94) | 1.88 (2.01) |
| efficientnet_b0 | 4.02 | 646 | 16.5 | 3.77 (27.77) | 2,422 | 1.61 (1.86) | 3.50 (3.72) |
| shufflenet_v2_x1_0 | 1.26 | 250 | 5.3 | 2.08 (9.52) | 7,351 | 0.81 (0.91) | 1.42 (1.56) |
