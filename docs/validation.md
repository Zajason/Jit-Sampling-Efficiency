# Setup validation

Validated on 5 November 2026 on the local Mac, using an isolated Python 3.11 environment.

## Passed checks

- Eight CPU unit tests passed (`python -m unittest discover -s tests -v`).
- Euler and Heun match the unmodified official `Denoiser.generate` exactly for 1, 2, 25 and 50 steps, using a small analytic network in place of pretrained weights. The actual upstream guidance and integration code executes in this test.
- Actual network forward counts match the expected budgets for all six pilot configurations; input noise is not modified.
- Variable-step AB2 integrates a linear-in-time velocity correctly from exact history.
- AB2 error on a smooth test ODE decreases by approximately 3.9569x when halving the step size, consistent with second-order convergence including Euler startup.
- Per-image noise is independent of batch partitioning; 2,000 labels contain exactly two occurrences of every ImageNet class.
- Official image quantization uses round-and-clip behavior, and nonfinite samples are rejected.
- Checkpoint loading preserves base buffers, applies first-EMA parameters, handles official argparse metadata with the restricted loader, and rejects absent/incomplete EMA weights.
- All downloaded official files, including the 32 MiB FID statistics file, match the pinned SHA-256 values. The bootstrap command verifies an already-downloaded snapshot without modifying it.
- Both experiment command-line entry points load and expose help.

## Provenance

- Official JiT: `cbc743a2ada5e9762697da2c83f8c4f8379e8c17` from `https://github.com/LTH14/JiT`.
- Optional official FID fork: `bfe72b995473e4fee9351705d7b005f17442a4ca` from `https://github.com/LTH14/torch-fidelity`.
- Test dependencies: torch 2.15.0, numpy 2.4.6, Pillow 13.0.0, einops 0.8.2. GPU execution records its own actual environment in each run manifest; the Mac test environment is not a CUDA installation recipe.

## Status at original setup (historical)

No pretrained checkpoint was downloaded or executed, and no image-generation benchmark, LPIPS calculation, or FID screen was run. The tests above establish code correctness on controlled inputs, not image quality or speedup. CUDA generation, checkpoint compatibility with the downloaded public artifact, GPU memory fit, and measured performance still require the RTX 5080 and the official checkpoint.

## CUDA execution — 5 November 2026

An isolated environment at `/mnt/research/jit-sampling-venv` uses Python 3.14.4,
PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130 and the RTX 5080 (capability 12.0).
CUDA tensor execution and bf16 support were checked. `pip check` passes.
The pinned official snapshot and FID statistics passed all SHA-256 checks.
The official JiT-B/16 checkpoint was downloaded through the pinned upstream
README's public Dropbox folder; its exact digest is in each run manifest.
The pinned FID fork and LPIPS feature weights are installed.

All 10 unit tests pass, including the existing exact uniform Euler/Heun parity
tests and added nonuniform-grid integration/invalid-schedule checks. These
remain analytic tests, distinct from the pretrained execution below.

The original 64-pair pretrained pilot completed at
`/mnt/research/jit-sampling/results/pilot-20261105-001`: 384 PNGs,
seven randomized timing rounds after two warm-ups, paired pixel RMSE and LPIPS.
Heun-50 median sampler time was 1091.41 ms/batch of four; AB2-25 was
275.11 ms/batch (3.967x measured speedup). These are amortized batch timings,
not batch-1 latency. The pilot has no FID and does not establish the 10%
quality target. Timing boundaries and complete provenance are in its manifest.

The larger study is tracked separately in
`/mnt/research/jit-sampling/results/study-20261105-001/status.json`.
Do not infer completion or quality results from this validation note.

The completed-stage audit subsequently verified 40,384 images across the pilot,
16-configuration uniform screen and four-configuration schedule validation,
including exact image IDs/counts, balanced screening labels, controlled
settings, actual forward counts, checkpoint consistency and pinned FID
provenance. All 10 tests still pass. See `docs/experiment-results.md` for the
measured tables and the NVIDIA UVM shutdown blocker that prevented starting
the two larger final evaluations.

## Project registration

The local project folder and Git repository were created. Adding the folder to the Codex sidebar requires the user's project picker: no project-creation connector is available, and computer use explicitly prohibits operating the Codex app itself. No application database or internal configuration was modified as a workaround.
