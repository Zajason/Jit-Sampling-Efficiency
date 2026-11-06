# Execution protocol — 5 November 2026

Scope: current proposal (`docs/jit-sampling-efficiency/proposal.pdf`). One official JiT-B/16 ImageNet-256 checkpoint, first EMA; no training. RTX 5080; bf16; batch 4; CFG 3.0, interval (0.1,1.0), t_eps 0.05; uniform official Heun-50 reference. All upstream hashes verified before each generation run. Environment and checkpoint digest are recorded per run.

1. Original 64-pair pilot: all six existing configurations, paired pixel RMSE and AlexNet LPIPS, seven randomized timing rounds after two warm-ups.
2. Uniform solver screen: Euler, Heun, AB2 at nominal 10,20,30,50,100 sample-equivalent network passes/image, plus Heun-50 (198 passes). Nearest feasible count at or below each budget, explicitly recorded. 2,000 balanced images/configuration; seed 20261106. Official pinned FID/IS evaluator and statistics.
3. Schedule validation: hold a solver and steps fixed; select the best screening FID configuration among the 50-pass methods, then compare t=u^p for p=0.5,1,2. Use 2,000 balanced images/configuration and seed 20271106. Freeze the lowest-FID schedule before final evaluation. Include Heun-50 for a paired reference.
4. Final evaluation: reference, selected uniform candidate, and selected schedule candidate (deduplicate). Uniform candidate is the fastest screen configuration within 1.10x screening reference FID; if none qualifies, use the lowest-FID nonreference configuration. Schedule candidate is selected only from validation results. Two independent runs of 5,000 balanced images/configuration, seeds 20281106 and 20291106. Report each FID and variation across the two runs; these sample sizes do not establish a population-level 10% guarantee.
5. Export per-run and combined quality/latency tables, Pareto frontier, paired images, and selected intermediate predictions for presentation use. Preserve negative/unstable results and all run outputs.

Final sample count chosen after initial measured reference throughput (~1.09 seconds per batch of four). It is a resource-limited evaluation, not the official 50,000-image protocol. All wall-clock estimates exclude setup and are estimates, not measured outcomes.

Timing boundary: synchronized sampler only; noise and labels already on GPU. Excludes loading, RNG, transfers, saving, metrics and untimed instrumentation. Distinguish solver steps, guided velocity evaluations, actual batch forward calls and sample-equivalent passes. Per-image time is amortized batch time. No compile or TF32. Peak allocated GPU memory includes the resident model and tensors.

Schedule tuning and final seeds are separate. Paired distances to generated Heun images measure agreement, not image quality. Screen FID is selection evidence only. All configs and the selection decision are saved before final generation.

Noise ranges are disjoint: screen 20261106–20263105; validation 20271106–20273105; final repeat 1 20281106–20286105; final repeat 2 20291106–20296105. This accounts for the harness using base_seed + image_id. These ranges were frozen before validation/final generation.
