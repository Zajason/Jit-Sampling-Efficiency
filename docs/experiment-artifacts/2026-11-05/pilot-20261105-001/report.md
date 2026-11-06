# JiT pilot results

Samples per configuration: 64; timing batch size: 4.

| Sampler | Passes/image | Batch ms | Amortized ms/image | Speedup | Paired pixel RMSE |
|---|---:|---:|---:|---:|---:|
| heun_50 | 198 | 1091.41 | 272.85 | 1.00x | 0.0000 |
| heun_25 | 98 | 547.66 | 136.92 | 1.99x | 0.0741 |
| euler_50 | 100 | 559.69 | 139.92 | 1.95x | 0.0700 |
| euler_25 | 50 | 279.10 | 69.77 | 3.91x | 0.1128 |
| ab2_50 | 100 | 553.80 | 138.45 | 1.97x | 0.0313 |
| ab2_25 | 50 | 275.11 | 68.78 | 3.97x | 0.0916 |

Timing excludes initial-noise creation and transfer, model loading, file saving and metrics.
Per-image time is amortized batch cost, not single-image latency unless batch size is 1.
Paired distances measure agreement with Heun-50, not image quality. No FID has been computed by this command.
Inspect paired_grid.png before selecting candidates for the separate 2000-image FID screen.
