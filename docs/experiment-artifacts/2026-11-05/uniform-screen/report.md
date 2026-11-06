# JiT pilot results

Samples per configuration: 2000; timing batch size: 4.

| Sampler | Passes/image | Batch ms | Amortized ms/image | Speedup | Paired pixel RMSE |
|---|---:|---:|---:|---:|---:|
| heun_50 | 198 | 1085.73 | 271.43 | 1.00x | 0.0000 |
| euler_5 | 10 | 55.95 | 13.99 | 19.40x | 0.1805 |
| euler_10 | 20 | 110.10 | 27.52 | 9.86x | 0.1583 |
| euler_15 | 30 | 160.43 | 40.11 | 6.77x | 0.1392 |
| euler_25 | 50 | 269.19 | 67.30 | 4.03x | 0.1109 |
| euler_50 | 100 | 541.84 | 135.46 | 2.00x | 0.0695 |
| heun_3 | 10 | 53.57 | 13.39 | 20.27x | 0.1816 |
| heun_5 | 18 | 99.03 | 24.76 | 10.96x | 0.1657 |
| heun_8 | 30 | 166.19 | 41.55 | 6.53x | 0.1485 |
| heun_13 | 50 | 273.09 | 68.27 | 3.98x | 0.1215 |
| heun_25 | 98 | 536.86 | 134.21 | 2.02x | 0.0741 |
| ab2_5 | 10 | 55.62 | 13.91 | 19.52x | 0.1719 |
| ab2_10 | 20 | 110.86 | 27.71 | 9.79x | 0.1455 |
| ab2_15 | 30 | 163.87 | 40.97 | 6.63x | 0.1240 |
| ab2_25 | 50 | 274.67 | 68.67 | 3.95x | 0.0902 |
| ab2_50 | 100 | 556.26 | 139.07 | 1.95x | 0.0344 |

Timing excludes initial-noise creation and transfer, model loading, file saving and metrics.
Per-image time is amortized batch cost, not single-image latency unless batch size is 1.
Paired distances measure agreement with Heun-50, not image quality. No FID has been computed by this command.
Inspect paired_grid.png before selecting candidates for the separate 2000-image FID screen.
