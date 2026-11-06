# JiT sampling efficiency

**Accelerating Just Image Transformers Through Efficient Numerical Sampling**  
Advanced Topics in Artificial Intelligence, NTUA ECE, 2026–2027.  
Iason-Daniil Zakynthinos.

This project tests whether existing numerical sampling methods can generate JiT images faster without retraining. The initial target is 2× lower sampling latency with FID within 10% of the reference. This is a hypothesis, not an achieved result.

## Proposal

- [Current two-page proposal](docs/jit-sampling-efficiency/proposal.pdf)
- [Editable proposal source](docs/jit-sampling-efficiency/build_proposal.py)
- [Earlier full revised draft](docs/jit-sampling-efficiency/original-revised-proposal.md)
- [Six-slide research presentation](docs/jit-sampling-efficiency/presentation.pptx)
- [Measured preliminary results](docs/experiment-results.md)
- [Archived result data, figures and paired grids](docs/experiment-artifacts/2026-11-05/README.md)

## What the pilot does

The pretrained network, guidance calculation, and Euler/Heun step functions come directly from the author-maintained [LTH14/JiT](https://github.com/LTH14/JiT) repository. A minimal upstream snapshot is pinned by commit and file hashes in `configs/upstream.json`; upstream files are never edited. This is the authors' PyTorch/GPU reimplementation; their original paper implementation used JAX/TPU.

The sampling loop for Adams–Bashforth 2 (AB2) is an explicitly identified addition. It starts with Euler and then reuses the previous velocity. All six original pilot configurations use uniform timesteps, first-EMA checkpoint weights, CFG 3.0 with interval (0.1, 1.0), and the upstream denominator clamp `t_eps=0.05`. The full proposal study below adds separately identified fixed nonuniform-grid ablations. Guidance is not optimized away outside its active interval, matching upstream compute behavior. The guidance switch and denominator clamp can make the dynamics nonsmooth; higher numerical order is therefore not a guarantee of better images.

| Configuration | Steps | Guided velocity evaluations | Network passes/image |
|---|---:|---:|---:|
| heun_50 (reference) | 50 | 99 | 198 |
| heun_25 | 25 | 49 | 98 |
| euler_50 | 50 | 50 | 100 |
| euler_25 | 25 | 25 | 50 |
| ab2_50 | 50 | 50 | 100 |
| ab2_25 | 25 | 25 | 50 |

A network pass here is a sample-equivalent evaluation of the transformer. A batch performs the same number of forward calls, each processing the batch's images. Actual calls are checked using an untimed forward hook, which is removed before benchmarking. Heun uses the upstream final Euler update.

The 64-image pilot produces 384 images and requires 38,144 sample-equivalent passes for sample generation, **plus** warm-up, timing repetitions, and call-count checks (61,984 passes including those overheads with the default settings). `python scripts/plan.py` prints this accounting without requiring a GPU or PyTorch.

## Setup on the CUDA machine

Use Python 3.10 or newer. On the RTX 5080, first install a CUDA-enabled PyTorch build supporting that GPU using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). The upstream model contains CUDA-specific attention code; CPU/MPS image generation is not supported by this pilot.

```bash
cd jit-sampling-efficiency
python3 -m venv .venv
source .venv/bin/activate
# Install the appropriate CUDA-enabled PyTorch build first.
python -m pip install -e .
python scripts/bootstrap_upstream.py
python -m unittest discover -s tests -v
```

Download the **JiT-B/16, ImageNet-256** pretrained checkpoint using the link in the [official README](https://github.com/LTH14/JiT#evaluation). Put the file under `checkpoints/`. The runner requires the official dictionary containing `model` and `model_ema1`; it loads base buffers and overlays the first EMA's parameters exactly as the official evaluator does. Missing weights are an error, never a fallback to random initialization. Checkpoints are loaded with PyTorch's restricted weights-only loader, allowing only the official argparse.Namespace metadata in addition to supported tensor containers.

No ImageNet training folder is required for generation: this harness bypasses the upstream training/distributed entry point and calls its actual denoiser and sampling functions directly.

## Run the initial experiment

```bash
jit-pilot \
  --checkpoint checkpoints/checkpoint-last.pth \
  --config configs/pilot.json \
  --output results/pilot-001
```

The output directory must be new. If memory is tight, copy the config and reduce `batch_size` (use 1 for single-image latency). Keep it identical across all configurations. Batch size 4 is only a starting point, not a verified memory-fit claim for the RTX 5080. Timing uses 2 warm-up runs and 7 randomized rounds per configuration, on the same fixed initial batch.

Noise is generated independently for each sample from its saved seed, so paired samples do not depend on batch boundaries. Labels are drawn from a deterministic permutation of all 1,000 classes; the 64-image pilot therefore spans 64 distinct classes. The 2,000-image screen gives exactly two images per class.

### Outputs

- `manifest.json`: checkpoint SHA-256, upstream revision/hashes, settings, software and GPU details; status becomes `complete` only when the run finishes.
- `pairs.json`: sample IDs, labels and noise seeds.
- `timings.json`: all timing repetitions, randomized orders and allocated-memory peaks.
- `summary.csv` and `report.md`: median and interquartile timing, speedup, memory and paired image distances.
- `samples/<configuration>/`: the generated 256×256 PNG images.
- `paired_grid.png`: the first eight paired outputs, with one column per configuration.

The timing boundary is **the sampler**, starting with noise and labels already on the GPU. It includes all transformer evaluations and solver work; it excludes initial-noise creation, transfers, loading, image saving and metric evaluation. Each measurement synchronizes the GPU before and after sampling. Per-image time is **amortized batch time**, not batch-1 latency unless batch size is 1. No compilation or TF32 acceleration is enabled. Peak memory is allocated memory including resident model/input/output tensors, not total device usage.

Pixel RMSE is provided as a cheap check of agreement with the reference. For paired perceptual agreement, install the optional LPIPS dependencies and rerun into a new output directory:

```bash
python -m pip install -e '.[perceptual]'
jit-pilot --checkpoint checkpoints/checkpoint-last.pth --output results/pilot-002 --lpips
```

LPIPS may download its pretrained feature weights on first use. Neither pixel RMSE nor LPIPS to a generated reference establishes image quality or distributional coverage. A reference image is not ground truth. Timing is completed before these metrics run.

## Follow-up: small FID screen

Inspect the paired grid first. Copy `configs/quality-screen.json` and select the reference plus two promising candidates. Its current `euler_50` and `ab2_50` entries are placeholders for a controlled comparison, not winners. Keep guidance, sample count, and reference statistics fixed.

```bash
jit-pilot \
  --checkpoint checkpoints/checkpoint-last.pth \
  --config configs/quality-screen.json \
  --output results/screen-001
python scripts/install_fid.py
jit-pilot-fid --run results/screen-001
```

The optional FID installer uses the pinned author-maintained `LTH14/torch-fidelity` fork because the stock PyPI evaluator does not expose the official `fid_statistics_file` interface. FID requires the pinned official `fid_stats/jit_in256_stats.npz`. The FID command checks complete generation, image IDs/counts and balanced labels, then writes `fid_screen.json` with evaluator provenance and the statistics checksum. It requests FID and Inception Score together, matching the official evaluator and avoiding the fork’s FID-only shortcut, which does not use the supplied statistics file. It refuses to overwrite an existing FID result.

**Interpretation:** 2,000-image FID is a screening measurement. It does not establish the proposal's 10% tolerance. Larger equal-count evaluations and repeatability analysis are needed before that claim. Report speedup and quality separately; fewer passes alone do not prove useful acceleration.

## Decision after the pilot

1. Are the low-cost outputs visually stable, and are they measurably faster?
2. Does AB2 improve on Euler at the same pass budget?
3. Does a candidate retain competitive screening FID at materially lower latency?

If only fewer-step Euler/Heun works, the project still yields a useful quality–latency comparison. If AB2 fails, report that rather than tuning on final test seeds. Nonuniform schedule experiments come after this baseline pilot.

## Full proposal study

`configs/solver-budgets.json` extends the original uniform pilot to nominal
10, 20, 30, 50 and 100 sample-equivalent network passes per image. Heun uses
the nearest feasible count at or below the budget; the exact mapping is saved
in the config. The official Heun-50 reference remains a separate 198-pass point.
Additional configurations explicitly record their method, steps and schedule
power. Nonuniform schedules use `t = u ** power` and the same official step
functions; they are identified ablations, not official sampler parity claims.

The sequential study runner performs the 2,000-image uniform screen, then
validation-only schedule selection with a fixed solver, and freezes finalists
before two independent 5,000-image evaluations. Read
[the execution protocol](docs/experiment-protocol.md) for the predeclared
selection rules, seeds, timing boundary and sample-size limitations.

```bash
python scripts/run_study.py \
  --checkpoint checkpoints/jit-b-16.pth \
  --output results/study-NEW
python scripts/export_study.py \
  --study results/study-NEW --pilot results/pilot-NEW \
  --output results/presentation-NEW
python scripts/capture_intermediates.py \
  --study results/study-NEW --checkpoint checkpoints/jit-b-16.pth \
  --output results/intermediates-NEW
```

Run the original LPIPS-enabled pilot first and inspect its paired grid.
The exporter additionally requires `matplotlib`; it creates PNG, SVG and PDF
figures, a consolidated CSV and presentation notes. Intermediate conditional
predictions are captured separately, outside all sampler timing.
Every output directory must be new. `status.json`, stage logs, per-stage
manifests and `selection.json` make progress and selection inspectable.

## Structure

```text
configs/                 Pilot, quality-screen and pinned upstream manifests
src/jit_pilot/           Official-model adapter, AB2, timing, images and FID entry points
scripts/                 Upstream bootstrap, cost plan and optional FID installer
tests/                   Numerical convergence, upstream parity, call counts and pairing
docs/jit-sampling-efficiency/  Proposal PDF and editable source
vendor/official-jit/     Downloaded official files; ignored by Git
checkpoints/             Downloaded pretrained checkpoint; ignored by Git
results/                 Actual experiment outputs; ignored by Git
```

The original project setup used analytic tests only. A pretrained 64-image
pilot was completed on the RTX 5080 on 5 November 2026; see
`docs/validation.md` for the distinction between those results and the setup
tests. The 16-configuration uniform screen and four-configuration schedule
validation also completed: [measured results and remaining work](docs/experiment-results.md).
Final evaluations await CUDA driver recovery. Larger study results must be
read from their completed run manifests.
