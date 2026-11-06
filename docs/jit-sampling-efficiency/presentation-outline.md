# Accelerating JiT — presentation content

Six slides, created 6 November 2026. Focus: question, literature, methodology and differentiation; preliminary numbers only on the last slide. Existing completed results retained; no further CUDA experiments launched.

Design: Presenton Sage Authority (14). Fonts: DM Serif Display and Manrope.

## Slide 1

Research question
Accelerating JiT through efficient numerical sampling
Iason-Daniil Zakynthinos · ECE, NTUA · Advanced Topics in AI · 2026–2027
How much sampling time can we remove from a pretrained pixel-space transformer while retaining image quality?
JiT predicts clean images directly from noisy pixels. Generation still requires repeated transformer evaluations along an ODE trajectory.
A fixed checkpoint lets us study the sampler as the independent variable.
H1 · Reuse velocity history
A multistep solver can offer a better quality–latency trade-off than repeated Heun corrections.
H2 · Reallocate timesteps
A fixed nonuniform grid can improve quality at the same solver and evaluation budget.
Research proposal · Inference-only study: unchanged architecture and pretrained weights. Motivation:
[1] Li & He, Back to Basics / JiT, arXiv:2511.13720
1/6

## Slide 2

Literature and positioning
The contribution is a controlled JiT study
Work
What it establishes
What this project investigates
JiT
[1]
Clean-image prediction with a transformer operating directly on pixels, without a latent tokenizer.
Keep JiT-B/16 fixed; isolate the quality cost of cheaper numerical sampling.
DPM-Solver++
[2]
Dedicated high-order guided diffusion solvers, including a multistep variant.
Test generic AB2 in JiT’s velocity formulation; do not assume the paper’s gains transfer.
Align Your Steps
[3]
Schedule optimization depends on the solver, trained model and dataset.
Separate timestep placement from solver choice using a small fixed-grid ablation.
Differentiating factor:
measured quality–latency frontiers for one JiT checkpoint, with realized network costs, official-sampler parity and held-out schedule selection. The contribution is empirical, rather than a new solver.
[1] Li & He, Back to Basics / JiT, arXiv:2511.13720
·
[2] Lu et al., DPM-Solver++, arXiv:2211.01095
·
[3] Sabour, Fidler & Kreis, Align Your Steps, ICML 2024
2/6

## Slide 3

Numerical method
One clean-image predictor, three integration rules
v(z,t) = [x̂(z,t) − z] / max(1 − t, ε)
JiT predicts the clean image x̂; the sampler constructs the velocity v and advances the noisy state z.
Euler
zₙ₊₁ = zₙ + hₙvₙ
One fresh velocity per step. A low-cost first-order baseline.
Heun
z* = zₙ + hₙvₙ
zₙ₊₁ = zₙ + ½hₙ(vₙ + v*)
A predictor and a fresh correction. Preserve the official final Euler update.
Adams–Bashforth 2
zₙ₊₁ = zₙ + ½h(3vₙ − vₙ₋₁)
Start with Euler; then reuse history with one fresh velocity. Use variable-step coefficients on nonuniform grids.
Why it is a real question:
guidance switches and the denominator clamp make the dynamics nonsmooth. Higher numerical order alone does not guarantee better images.
hₙ = tₙ₊₁ − tₙ; v* is evaluated at (z*, tₙ₊₁). AB2 formula shown for uniform steps. Official velocity/steps:
LTH14/JiT
.
3/6

## Slide 4

Experimental design
Separate solver choice from timestep placement
A · Solver comparison
Euler, Heun and AB2 on uniform grids.
Nominal budgets: 10, 20, 30, 50 and 100 sample-equivalent passes per image. Report the nearest feasible realized count.
B · Schedule ablation
Hold the solver and pass budget fixed.
Compare t = uᵖ with p ∈ {½, 1, 2}. Select on separate validation noise; freeze the grid before final evaluation.
C · Held-out evaluation
Compare frozen finalists with the official Heun reference.
Use larger, equal-sized image sets and independent seed ranges. Report repeatability and the quality–latency frontier.
Controls shared across every comparison
JiT-B/16, ImageNet-256, first EMA · same paired noise and labels · CFG 3.0 · fixed guidance interval and endpoint clamp · bf16 · batch 4 · RTX 5080
Pinned official files remain unchanged. Euler/Heun parity is checked analytically; AB2 and nonuniform schedules are explicitly identified additions.
Protocol: configs/solver-budgets.json and docs/experiment-protocol.md. Separate and disjoint noise ranges prevent tuning on final evaluation seeds.
4/6

## Slide 5

Evaluation and interpretation
Test the trade-off, not just the step count
Evidence needed
Distributional quality
FID against fixed ImageNet statistics; equal image counts and preprocessing. Paired LPIPS and grids explain agreement, not distributional quality.
Measured computational cost
Synchronized sampler latency after warm-up; randomized repeated rounds, median/IQR and peak allocated memory.
Timing excludes loading, noise creation/transfers, image saving and metrics. Per-image time is amortized batch cost.
What would support the hypotheses?
H1:
AB2 offers lower FID at matched realized cost, or comparable FID with lower latency.
H2:
a validation-selected nonuniform grid improves held-out quality at the same solver and budget.
Objective: at least 2× faster, with FID ≤ 1.10× the reference.
A target to test. Negative results still map useful operating points and the limits of reuse.
Count solver steps, guided velocity evaluations, actual batch forward calls and sample-equivalent passes separately. Fewer passes alone are not measured speedup.
5/6

## Slide 6

Preliminary evidence · not final results
Preliminary evidence justifies a larger test
Uniform-grid screen · 2,000 balanced images per configuration · same seeds and labels
Sampler
Passes / image
ms / image¹
Speedup
Screening FID ↓
Heun-50 reference
198
271.43
1.00×
19.223
Euler-25
50
67.30
4.03×
20.404
AB2-25
50
68.67
3.95×
19.557
Encouraging for H1
AB2 has lower observed FID than Euler at the same pass budget and similar latency.
No support yet for H2
Neither tested nonuniform grid beat uniform AB2 on separate validation seeds.
Small-sample FID is selection evidence. Final held-out evaluations are deferred; the quality-tolerance target is not established.
¹ Amortized batch-4 sampler time, bf16, RTX 5080; excludes I/O and metrics. Source: completed uniform-screen and schedule-validation runs, 5 Nov 2026; docs/experiment-results.md.
6/6

## Sources

- Current proposal: proposal.pdf and build_proposal.py in this directory.
- Measured results: ../experiment-results.md
- https://arxiv.org/abs/2511.13720
- https://arxiv.org/abs/2211.01095
- https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/
- https://github.com/LTH14/JiT
