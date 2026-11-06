# JiT sampling experiments — completed evidence and remaining work

Status: pilot, uniform solver screen and schedule validation completed. Final evaluations have not run.

**40,384 pretrained images generated.** All dependencies were installed in an isolated CUDA environment; all 10 unit tests pass. Official upstream files and FID statistics match their pinned hashes.

## Main screening findings

- Euler-25: 4.03x measured sampler speedup; 2,000-image FID 20.404 versus 19.223 for Heun-50 (approximately 6.1% higher). This is screening evidence, not confirmation of the 10% quality tolerance.
- AB2-25: screening FID 19.557 at 3.95x speedup. At the same 50-pass budget, it has lower observed screening FID than Euler-25.
- Neither tested nonuniform schedule improves AB2-25 validation FID. Uniform: 19.439; square-root schedule: 28.925; squared schedule: 20.651. This finding applies to these two tested schedules.

## Controlled setup

Checkpoint SHA-256: `4ebcf24698748548d13bef1b4c3b26c72c6ec2bc633002b3f558697920cb2695`. Official JiT-B/16, ImageNet-256, first EMA; RTX 5080; Python 3.14.4; PyTorch 2.14.1+cu130; bf16; batch 4; CFG 3.0; interval (0.1,1.0); t_eps=0.05; no training, compilation or TF32. Labels, initial noise and settings are paired within each comparison. Validation and final noise ranges are disjoint.

Timing is synchronized sampler time with initial noise and labels already on GPU. It excludes model loading, RNG, transfers, warm-up, call-count checks, saving, metrics and interpreter cleanup. Per-image times below are amortized batch times, not batch-1 latency. Seven randomized rounds follow two warm-ups. Full repetitions, IQRs and peak allocated memory are in each summary.csv and timings.json.

Guided velocity evaluations, actual network forward calls per batch, and sample-equivalent passes/image are recorded separately. Here each guided evaluation performs two actual batch calls. Actual calls/batch therefore equal passes/image; total sample-equivalent work per batch is four times that count. The reference has 50 solver steps, 99 guided evaluations, 198 actual calls/batch and 198 passes/image. Heun retains its final Euler update. AB2 and nonuniform schedules are explicit additions.

## Uniform solver screen

2,000 balanced images/configuration, seed 20261106. FID uses the pinned official evaluator and ImageNet reference statistics.

| Configuration | Passes/image | Amortized ms/image | Speedup | Screening FID |
|---|---:|---:|---:|---:|
| heun_50 | 198 | 271.43 | 1.000x | 19.223 |
| euler_5 | 10 | 13.99 | 19.404x | 37.515 |
| euler_10 | 20 | 27.52 | 9.861x | 26.279 |
| euler_15 | 30 | 40.11 | 6.768x | 23.489 |
| euler_25 | 50 | 67.30 | 4.033x | 20.404 |
| euler_50 | 100 | 135.46 | 2.004x | 19.388 |
| heun_3 | 10 | 13.39 | 20.266x | 41.075 |
| heun_5 | 18 | 24.76 | 10.964x | 25.448 |
| heun_8 | 30 | 41.55 | 6.533x | 23.153 |
| heun_13 | 50 | 68.27 | 3.976x | 21.881 |
| heun_25 | 98 | 134.21 | 2.022x | 19.462 |
| ab2_5 | 10 | 13.91 | 19.520x | 30.506 |
| ab2_10 | 20 | 27.71 | 9.794x | 23.808 |
| ab2_15 | 30 | 40.97 | 6.626x | 21.726 |
| ab2_25 | 50 | 68.67 | 3.953x | 19.557 |
| ab2_50 | 100 | 139.07 | 1.952x | 19.294 |

## Schedule validation

2,000 balanced images/configuration, disjoint seed 20271106. AB2 solver and 25 steps / 50 passes are fixed.

| Configuration | Amortized ms/image | Validation FID |
|---|---:|---:|
| heun_50 | 268.65 | 19.288 |
| ab2_25 | 67.62 | 19.439 |
| ab2_25_p0_5 | 68.40 | 28.925 |
| ab2_25_p2_0 | 67.80 | 20.651 |

## Pilot agreement

64 paired images/configuration; seed 20261105. LPIPS to a generated reference measures agreement, not image quality.

| Configuration | Amortized ms/image | Paired LPIPS |
|---|---:|---:|
| heun_50 | 272.85 | 0.0000 |
| heun_25 | 136.92 | 0.1909 |
| euler_50 | 139.92 | 0.1768 |
| euler_25 | 69.77 | 0.3132 |
| ab2_50 | 138.45 | 0.0549 |
| ab2_25 | 68.78 | 0.2436 |

## CUDA recovery and remaining evaluations

After validation generation and FID had saved their complete outputs, CUDA event-handler threads remained in uninterruptible NVIDIA UVM kernel waits (`uvm_parent_gpu_replayable_faults`). Termination did not fully clear those processes. No final benchmarks were started in that state. Completed artifacts, hashes, metric results and sampler timings were preserved; cleanup incidents are recorded in cleanup-events.jsonl. The desktop also uses this GPU, so no disruptive GPU reset or reboot was attempted.

After a user-controlled reboot and a successful CUDA smoke check, resume the frozen finalists: Heun-50, Euler-25, AB2-25. Two independent 5,000-image repeats use bases 20281106 and 20291106. The selected configuration list and decision were saved before any final generation. Untimed intermediate prediction grids remain pending too.

```bash
TORCH_HOME=/mnt/research/jit-sampling/torch-cache OMP_NUM_THREADS=8 \
  .venv/bin/python scripts/run_study.py \
  --checkpoint checkpoints/jit-b-16.pth \
  --output results/study-20261105-001 --resume
```

Reuse only complete matching stages. Partial generation is never overwritten. Final sample sizes are resource-limited; report each final FID and variability rather than asserting a population-level tolerance from screening results.

## Presentation assets

The [GitHub result archive](experiment-artifacts/2026-11-05/README.md) contains measured FID scores, all timing repetitions, summary tables, paired seeds/labels, manifests, figures and paired image grids. The full raw image corpus and checkpoint remain outside Git.

The [six-slide PowerPoint](jit-sampling-efficiency/presentation.pptx) was subsequently created from the proposal and completed results. It focuses on methodology, literature and hypotheses, with one slide of preliminary numbers. Its [editable content outline](jit-sampling-efficiency/presentation-outline.md) is also tracked.

Local originals remain at `/mnt/research/jit-sampling/results/presentation-20261105-002/` and in each generation run.

Source snapshots, manifests, checkpoint/evaluator provenance, settings, labels/noise seeds and the completed-stage audit are preserved under `/mnt/research/jit-sampling/results/study-20261105-001/`.
