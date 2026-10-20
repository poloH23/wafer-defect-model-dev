Mean of the last 5 epochs per run; mean ± standard deviation over runs (seeds). FLOPs per image.

| Config | Seeds | Params (M) | MFLOPs | val_errors | val_exact_match | val_macro_f1 | val_macro_ap |
|---|---|---|---|---|---|---|---|
| wafer_augment (learning_rate=0.0003) | 0 | 11.17 | 3047 | 47.4 | 0.9872 | 0.9838 | 0.9971 |
| wafer_augment (learning_rate=0.003) | 0 | 11.17 | 3047 | 44.6 | 0.9879 | 0.9864 | 0.9982 |
| wafer_augment | 0, 1, 2 | 11.17 | - | 44.1 ± 4.2 | 0.9881 ± 0.0011 | 0.9849 ± 0.0007 | 0.9977 ± 0.0003 |
| wafer_mobilenet_v3_small (learning_rate=0.0003) | 0 | 1.53 | 94 | 54.2 | 0.9854 | 0.9843 | 0.9983 |
| wafer_mobilenet_v3_small (learning_rate=0.003) | 0 | 1.53 | 94 | 43.6 | 0.9882 | 0.9923 | 0.9984 |
| wafer_mobilenet_v3_small | 0, 1, 2 | 1.53 | 94 | 46.2 ± 0.2 | 0.9875 ± 0.0001 | 0.9873 ± 0.0045 | 0.9979 ± 0.0012 |
| wafer_mobilenet_v3_large (learning_rate=0.0003) | 0 | 4.21 | 355 | 51.0 | 0.9862 | 0.9915 | 0.9991 |
| wafer_mobilenet_v3_large (learning_rate=0.003) | 0 | 4.21 | 355 | 39.2 | 0.9894 | 0.9926 | 0.9993 |
| wafer_mobilenet_v3_large | 0, 1, 2 | 4.21 | 355 | 43.9 ± 1.0 | 0.9881 ± 0.0003 | 0.9872 ± 0.0007 | 0.9982 ± 0.0001 |
| wafer_efficientnet_b0 (learning_rate=0.0003) | 0 | 4.02 | 646 | 49.0 | 0.9868 | 0.9826 | 0.9981 |
| wafer_efficientnet_b0 (learning_rate=0.003) | 0 | 4.02 | 646 | 37.0 | 0.9900 | 0.9871 | 0.9973 |
| wafer_efficientnet_b0 | 0, 1, 2 | 4.02 | 646 | 42.1 ± 2.0 | 0.9886 ± 0.0005 | 0.9868 ± 0.0010 | 0.9973 ± 0.0004 |
| wafer_shufflenet_v2_x1_0 (learning_rate=0.0003) | 0 | 1.26 | 250 | 81.2 | 0.9781 | 0.9847 | 0.9963 |
| wafer_shufflenet_v2_x1_0 (learning_rate=0.003) | 0, 1, 2 | 1.26 | 250 | 43.1 ± 2.1 | 0.9884 ± 0.0006 | 0.9918 ± 0.0024 | 0.9987 ± 0.0012 |
| wafer_shufflenet_v2_x1_0 (learning_rate=0.01) | 0 | 1.26 | 250 | 44.8 | 0.9879 | 0.9915 | 0.9993 |
| wafer_shufflenet_v2_x1_0 | 0 | 1.26 | 250 | 50.2 | 0.9864 | 0.9907 | 0.9988 |
| wafer_slim_resnet_v1_t015 | 0, 1, 2 | 0.70 | 18 | 71.7 ± 9.8 | 0.9806 ± 0.0027 | 0.9811 ± 0.0105 | 0.9945 ± 0.0020 |
| wafer_slim_resnet_v1_t046 | 0, 1, 2 | 0.69 | 20 | 64.1 ± 9.1 | 0.9827 ± 0.0025 | 0.9860 ± 0.0054 | 0.9959 ± 0.0005 |
| wafer_slim_resnet_v1_t010 | 0, 1, 2 | 1.23 | 33 | 58.7 ± 5.3 | 0.9841 ± 0.0014 | 0.9834 ± 0.0022 | 0.9962 ± 0.0010 |
| wafer_slim_resnet_v1_t029 | 0, 1, 2 | 0.70 | 192 | 48.7 ± 1.1 | 0.9868 ± 0.0003 | 0.9852 ± 0.0012 | 0.9971 ± 0.0010 |
| wafer_slim_resnet_v1_t036 | 0, 1, 2 | 1.23 | 349 | 53.1 ± 6.1 | 0.9856 ± 0.0017 | 0.9886 ± 0.0028 | 0.9989 ± 0.0005 |
