# Measured JiT experiment results

All charts use actual pretrained JiT-B/16 outputs. No training.

Timing: synchronized sampler only; excludes model loading, noise generation/transfers, warm-up, call-count checks, image saving and metrics. Batch 4 bf16 on RTX 5080; per-image times are amortized, not single-image latency.

Pilot: 64 paired images/configuration. LPIPS and pixel RMSE measure agreement with generated Heun-50, not distributional quality.

Screen/validation: 2,000 balanced images/configuration. Schedule tuning uses separate validation seeds. Final protocol: two independent seeds, 5,000 balanced images/configuration/repeat. FID variability from two repeats is descriptive; no population-level 10% claim.

**Final evaluations have not run.** CUDA driver recovery is required. The completed screen and validation results below do not establish the proposal target.


## Matched-pass AB2 versus Euler (screening only)

- 10 passes/image: Euler FID 37.515; AB2 FID 30.506.
- 20 passes/image: Euler FID 26.279; AB2 FID 23.808.
- 30 passes/image: Euler FID 23.489; AB2 FID 21.726.
- 50 passes/image: Euler FID 20.404; AB2 FID 19.557.
- 100 passes/image: Euler FID 19.388; AB2 FID 19.294.

## Schedule validation (separate seeds)

Fixed solver ab2_25; uniform FID 19.439. Lowest validation FID: ab2_25, 19.439. Selection uses these validation results only.

Full settings, exact labels/noise seeds, checkpoint/source hashes, timing repetitions and evaluator provenance are saved in each run directory.
PNG figures are suitable for slides; SVG/PDF versions are provided for resizing.
