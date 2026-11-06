# JiT pilot results

Samples per configuration: 2000; timing batch size: 4.

| Sampler | Passes/image | Batch ms | Amortized ms/image | Speedup | Paired pixel RMSE |
|---|---:|---:|---:|---:|---:|
| heun_50 | 198 | 1074.59 | 268.65 | 1.00x | 0.0000 |
| ab2_25 | 50 | 270.48 | 67.62 | 3.97x | 0.0908 |
| ab2_25_p0_5 | 50 | 273.59 | 68.40 | 3.93x | 0.1752 |
| ab2_25_p2_0 | 50 | 271.19 | 67.80 | 3.96x | 0.0836 |

Timing excludes initial-noise creation and transfer, model loading, file saving and metrics.
Per-image time is amortized batch cost, not single-image latency unless batch size is 1.
Paired distances measure agreement with Heun-50, not image quality. No FID has been computed by this command.
Inspect paired_grid.png before selecting candidates for the separate 2000-image FID screen.
