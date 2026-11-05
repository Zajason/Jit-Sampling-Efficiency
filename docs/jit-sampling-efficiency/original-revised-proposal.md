# Accelerating Just Image Transformers Through Efficient Numerical Sampling

**Advanced Topics in Artificial Intelligence — Academic Year 2026–2027**  
**School of Electrical and Computer Engineering, National Technical University of Athens**

**Student:** Iason-Daniil Zakynthinos

---

## 1. Abstract

Just image Transformers (JiT) demonstrate that high-quality image generation can be performed directly in pixel space using a transformer that predicts clean images from noisy inputs. However, generating an image requires repeatedly evaluating the neural network along a numerical sampling trajectory, making inference computationally expensive.

This project investigates whether JiT inference can be substantially accelerated by replacing its default sampling procedure with more computationally efficient numerical methods, without modifying or retraining the underlying neural network.

Specifically, we will compare first-order Euler integration, second-order Heun integration, and second-order multistep methods under controlled computational budgets. We will also investigate whether nonuniform timestep schedules can further improve sampling efficiency.

The primary objective is to identify the sampling configuration that offers the best trade-off between image quality and inference latency, with an initial target of achieving a **2× reduction in generation time while maintaining image quality close to the reference configuration**.

The project will produce a reproducible evaluation of sampling efficiency in JiT, including quantitative quality metrics, runtime measurements, and visual comparisons of generated images.

## 2. Background and Motivation

Recent generative models commonly formulate image generation as the numerical integration of a learned continuous-time process that transforms random noise into realistic images.

Just image Transformers, introduced by Li and He, adopt a particularly simple approach: a transformer directly predicts the clean image in pixel space, avoiding the need for a separately trained latent autoencoder.

Although the architecture is conceptually simple, inference still involves numerous neural network evaluations.

JiT's reference sampling setup uses numerical methods such as Euler and Heun to approximate the generation trajectory. Heun generally improves numerical accuracy by evaluating the model more than once per integration step, but this increases computational cost.

An alternative is to use multistep numerical methods, which reuse information from previous evaluations to construct higher-order updates without requiring multiple new network evaluations per step.

Furthermore, uniform timestep spacing may allocate network evaluations inefficiently across different parts of the generation trajectory.

These observations motivate a practical question: **Can we generate images of comparable quality using substantially fewer neural network evaluations?**

This project focuses exclusively on inference-time optimization, leaving the pretrained model and its parameters unchanged.

## 3. Research Question and Hypothesis

### Primary research question

**How much can the inference time of a pretrained Just image Transformer be reduced through efficient numerical solvers and timestep scheduling, without significantly degrading generated image quality?**

### Main hypothesis

A second-order multistep sampling method, potentially combined with a nonuniform timestep schedule, can achieve a more favorable quality–latency trade-off than the reference Heun sampler.

### Secondary questions

1. How do Euler, Heun, and second-order multistep methods compare when given the same number of neural network evaluations?
2. Does nonuniform timestep spacing improve image quality under a fixed computational budget?
3. Which sampling configuration offers the best balance of image quality, generation time, and computational cost?

The project does not aim to introduce a fundamentally new numerical solver. Its contribution is a controlled experimental evaluation of existing methods in the context of pixel-space image generation with JiT.

## 4. Proposed Methodology

### 4.1 Model and experimental environment

We will use the publicly available pretrained **JiT-B/16** model and generate images at a resolution of 256 × 256 pixels.

The experiments will be conducted using PyTorch on an NVIDIA RTX 5080 GPU, subject to initial compatibility and memory tests.

The model weights, architecture, and training configuration will remain unchanged throughout the project.

All comparisons will use consistent class labels, random seeds, guidance settings, and evaluation procedures.

### 4.2 Numerical sampling methods

We will implement and evaluate three sampling approaches.

**A. First-order Euler method**

Euler integration provides the simplest baseline, requiring one velocity evaluation for each integration step:

\[
z_{n+1}=z_n+h_nv_n
\]

where \(z_n\) is the current noisy image representation, \(h_n\) is the timestep interval, and \(v_n\) is the velocity predicted from JiT's clean-image estimate.

Euler is computationally inexpensive but may introduce substantial numerical error when large steps are used.

**B. Second-order Heun method**

Heun integration uses an initial velocity estimate and a second evaluation to correct the update:

\[
z_{n+1}=z_n+\frac{h_n}{2}(v_n+v_{n+1}^{*})
\]

where \(v_{n+1}^{*}\) is evaluated at an Euler-predicted intermediate state.

This method provides a reference for assessing whether improved numerical accuracy justifies additional neural network evaluations.

**C. Second-order multistep method**

We will investigate a two-step method based on Adams–Bashforth integration.

For constant timesteps:

\[
z_{n+1}=z_n+\frac{h}{2}(3v_n-v_{n-1})
\]

Unlike Heun, this method reuses the velocity from the preceding step, requiring only one new network evaluation per subsequent update.

For nonuniform timestep schedules, we will use the corresponding variable-step coefficients.

The hypothesis is that this approach can retain some of the numerical advantages of second-order integration while reducing the number of expensive model evaluations.

### 4.3 Timestep scheduling

We will compare two timestep allocation strategies:

- **Uniform scheduling:** Integration steps are distributed evenly across the sampling interval.
- **Nonuniform scheduling:** Steps are concentrated in selected regions of the trajectory using a parameterized schedule.

Nonuniform schedules will be tuned using a separate validation set.

This experiment will determine whether improvements can be obtained simply by changing where network evaluations occur, independently of the numerical solver.

Unlike the original project proposal, we will not implement image-specific adaptive timestep selection.

### 4.4 Experimental design

Sampling configurations will be evaluated at the following network-evaluation budgets:

| Budget | Purpose |
|---|---|
| 10 evaluations | Extreme low-cost sampling |
| 20 evaluations | Aggressive acceleration |
| 30 evaluations | Intermediate operating point |
| 50 evaluations | Moderate-cost sampling |
| 100 evaluations | Higher-quality reference range |

The budgets will count actual neural network evaluations, including any additional evaluations required by classifier-free guidance or multistage solvers.

For each sampling configuration, we will record image quality and end-to-end inference latency.

The experiments will be organized to isolate three effects:

1. **Solver efficiency:** Compare solvers under the same evaluation budget and uniform timestep spacing.
2. **Schedule efficiency:** Compare uniform and nonuniform schedules while keeping the solver unchanged.
3. **Combined optimization:** Identify the best-performing solver–schedule combination and compare it against the reference JiT sampling configuration.

We will initially conduct experiments on smaller image sets to identify promising configurations, followed by larger-scale evaluation of selected methods.

## 5. Evaluation Metrics

The primary metrics will be:

**Fréchet Inception Distance (FID).** Measures the statistical similarity between generated images and real images. Lower values indicate better alignment with the reference image distribution. Final FID comparisons will use consistent sample counts and evaluation settings.

**Inference latency.** Measures the wall-clock time required to generate an image, accounting for all model evaluations and sampling operations.

**Number of Network Function Evaluations (NFE).** Measures the principal computational cost of numerical sampling independently of the particular solver's step count.

**Peak GPU memory usage.** Evaluates whether alternative sampling procedures introduce additional memory requirements.

Additionally, selected generated images will be compared visually using identical initial noise and class labels.

Intermediate clean-image predictions may also be visualized to illustrate how differences between sampling methods affect the generation trajectory.

The final results will be summarized through quality–latency curves, allowing direct identification of Pareto-efficient sampling configurations.

## 6. Expected Results and Contributions

The expected outcome is a systematic comparison of numerical sampling strategies for Just image Transformers.

We aim to determine whether an appropriately selected solver and timestep schedule can reduce inference latency by approximately **2× while retaining FID within 10% of the reference sampler**, under the same evaluation protocol.

This is an experimental target rather than an assumed result.

Even if this target is not achieved, the project will provide a useful characterization of the relationship between sampling cost, numerical method, and generated image quality.

The main deliverables will be:

1. An implementation of alternative JiT sampling methods.
2. A controlled evaluation across multiple computational budgets.
3. A quantitative comparison of inference latency and image quality.
4. A visual comparison of generated images and intermediate predictions.
5. A final technical report documenting the methodology, findings, and limitations.

The project emphasizes reproducibility, computational efficiency, and experimental analysis rather than architectural modifications or model training.

## 7. Implementation Plan

| Weeks | Activities |
|---|---|
| 1–2 | Set up JiT, load pretrained weights, reproduce reference generation results, benchmark hardware performance |
| 3–4 | Implement and validate Euler, Heun, and second-order multistep sampling |
| 5–6 | Compare solvers under fixed network-evaluation budgets |
| 7–8 | Implement nonuniform timestep schedules and perform validation-set tuning |
| 9–10 | Evaluate promising configurations using larger generated image sets |
| 11 | Analyze quality–latency trade-offs and produce visual comparisons |
| 12–13 | Prepare final report, reproducible code, and presentation |

The scope is deliberately limited to a single pretrained model and inference-only modifications.

If computational constraints prevent exhaustive evaluation, the study will prioritize solver comparisons and reduce the number of timestep schedules and evaluation budgets.

## 8. Related Work

**Li and He — Just image Transformers.** Introduces the pixel-space image generation framework and pretrained model investigated in this project.

**Lu et al. — DPM-Solver++.** Demonstrates how specialized high-order numerical solvers can accelerate diffusion-model sampling, providing motivation for studying solver efficiency.

**Sabour et al. — Align Your Steps.** Investigates optimized timestep schedules for diffusion models, motivating the comparison between uniform and nonuniform sampling.

These works establish that numerical integration and timestep allocation are important factors in generative inference efficiency.

Our project examines how these ideas perform in JiT under directly comparable computational and latency budgets.

## 9. Conclusion

This project investigates a practical approach to accelerating a pretrained pixel-space generative model without additional training.

By comparing numerical solvers and timestep schedules, we aim to identify how much computational cost can be eliminated from JiT inference while preserving generated image quality.

The study combines numerical methods, deep generative modeling, and inference optimization, making it relevant to both advanced artificial intelligence research and efficient AI systems.

The intended outcome is a clear, reproducible understanding of the image-quality cost of faster JiT generation, along with a recommended sampling configuration for different computational budgets.
