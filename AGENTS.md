# Project guidance

This is a semester research project, not a new-model training project. Read README.md and configs/upstream.json before changing experiment behavior.

- Keep official JiT files under vendor/official-jit unchanged. Verify their pinned hashes before a run.
- Euler/Heun must retain parity with the official sampler. AB2 is an explicitly identified addition.
- Never label analytic/mock-network tests as pretrained generation results. Never invent timings, FID or speedups.
- Keep guidance, seeds, labels, precision, batch size and sample counts controlled and recorded.
- Keep warm-up, image saving and metric calculations out of sampler timing. State timing boundaries.
- Distinguish solver steps, guided velocity evaluations, actual forward calls and sample-equivalent network passes.
- Preserve existing results; use a new output directory for each run.
- Test with `python -m unittest discover -s tests -v`. GPU benchmark execution requires an appropriate CUDA environment and official checkpoint.
- The current proposal is docs/jit-sampling-efficiency/proposal.pdf. Its editable source is beside it.
