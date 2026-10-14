# Iterative CUDA optimisation and a roofline-aware reward, built on KernelGYM

My research extension of [KernelGYM](https://github.com/hkust-nlp/KernelGYM), from my Mitacs
Globalink internship at the University of Toronto (supervisor: Prof. Maryam Mehri Dehnavi).
It studies **LLM-driven GPU kernel optimisation**: how to train a model to keep improving a
CUDA kernel over several turns instead of stopping as soon as it has something that works.

> **Credit.** KernelGYM, the Dr. Kernel training method and the `drkernel-14b` model are by
> HKUST-NLP ([paper, arXiv:2602.05885](https://arxiv.org/abs/2602.05885)), licensed Apache-2.0.
> I did not write them. Their README is preserved in [UPSTREAM_README.md](UPSTREAM_README.md).
> This page covers only what I added; every file is listed in [CONTRIBUTIONS.md](CONTRIBUTIONS.md).

## The problem

In multi-turn kernel optimisation, the model writes a kernel, sees the evaluation, and
decides whether to try again. A reward based only on the final kernel being correct is the
simplest choice. But it makes stopping immediately the safest strategy: any further edit
risks breaking a correct kernel and gains nothing. The model learns not to optimise.

Using raw speedup instead has its own flaw. Speedup over a reference doesn't say how much
room for improvement was left, so it is a poor measure of real progress.

## What I built

### 1. A multi-turn CUDA optimisation workflow (`cuda_iterative`)

A new workflow plugged into KernelGYM's registries, in which the model iterates on a kernel and
emits **CONTINUE** or **STOP** each turn.

| Part | File |
|---|---|
| Workflow controller: turn loop, STOP detection, turn history, `max_turns` cap | `kernelgym/workflow/cuda_iterative_optimize.py` |
| Task and trajectory schema | `kernelgym/schema/cuda_iterative_task.py` |
| Backend: compiles and loads `ModelNew` CUDA kernels | `kernelgym/backend/cuda_iterative.py` |
| Toolkit: correctness and timing against the turn-0 baseline | `kernelgym/toolkit/cuda_iterative/` |
| Baseline rewards (terminal-correctness and per-turn), prompt builders, and the Dr. Kernel paper's TRLOO advantages and rejection sampling re-implemented for this workflow | `drkernel/kernel/rewards/iterative_cuda_reward.py` |

### 2. A roofline-aware terminal reward

`drkernel/kernel/rewards/roofline_reward.py`. It scores a kernel by how close it gets to the
**hardware limit**, using the roofline model (Williams, Waterman and Patterson, 2009):

```
arithmetic intensity  AI      = FLOPs / bytes moved
roofline ceiling              = min(peak_FLOP/s, peak_bandwidth × AI)
roofline efficiency           = achieved FLOP/s / ceiling          (clipped to [0, 1])

reward  R = C_final × clip(efficiency_final − efficiency_turn0, 0, 1)
```

- `C_final` is 1 only if the final kernel is correct, so a broken kernel always scores 0.
- Only improvement over the turn-0 kernel counts, so getting worse scores 0, not a penalty.
- Efficiency is measured against what the GPU could achieve at that kernel's intensity, so a
  memory-bound kernel can't game the signal by doing less work.
- Peak-performance figures for A100, H100, RTX 3060 Laptop and RTX 5090 are in a lookup table.
  An unknown GPU raises an error instead of guessing.
- When FLOPs or byte counts are missing, it falls back to correctness-gated speedup and logs a
  warning with the problem id, so the fallback is never silent.
- It is deliberately separate from the baseline reward code, and works without torch or a GPU.

**14 unit tests** cover:

- memory-bound and compute-bound efficiency;
- clipping at the ceiling;
- regressions and incorrect kernels;
- GPU lookup;
- the fallback path.

### 3. Running it on a GPU cluster (Trillium)

| What | Files |
|---|---|
| Container images | `kernelgym-cuda-trillium.def` (Apptainer), `Dockerfile.trillium`, `Dockerfile.smoke` |
| SLURM job for evaluating `drkernel-14b` on one H100 | `scripts/run_drkernel14b_trillium.sh` |

Fixes needed to make it work on Trillium:

- Compute nodes have no internet, so Hugging Face runs in offline mode.
- `HF_TOKEN` is read from a file, because SLURM doesn't pass the environment through.
- `SSL_CERT_FILE` is set inside the container.
- `CUDA_HOME` can be overridden.
- The partition and GPU directives match the cluster's scheduler.

### 4. Evaluation harness

| What | Files |
|---|---|
| Baseline run of `drkernel-14b` on the new workflow over 8 reference problems, with `--smoke` and `--dry-run` modes | `scripts/eval_real/` |
| Smoke tests for the backend and workflow that need no model | `scripts/eval_smoke/` |
| Error inspection | `scripts/show_errors.py` |

I rewrote the 8 reference kernels to use C++ launchers, so they compile through
`load_inline`.

## Status

The workflow, the rewards and the cluster setup are implemented. The baseline evaluation
harness runs end to end. **No reinforcement-learning results are reported in this repository
yet.** The roofline reward is the training signal under study, and no figures are claimed
until training runs are complete.

## Try it

```bash
pip install pytest
python -m pytest -q tests                                            # 14 roofline reward tests, CPU only
python scripts/eval_real/run_drkernel14b_cuda_iterative.py --smoke --dry-run   # needs the KernelGYM deps
```

Running the full environment (API server, GPU workers, Redis) follows the upstream setup in
[UPSTREAM_README.md](UPSTREAM_README.md).

## License

Apache-2.0, the same as upstream (see [LICENSE](LICENSE)). The original KernelGYM code is
© its authors; my additions are listed in [CONTRIBUTIONS.md](CONTRIBUTIONS.md).
