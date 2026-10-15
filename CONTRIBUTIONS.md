# Contributions in this fork

Upstream: [hkust-nlp/KernelGYM](https://github.com/hkust-nlp/KernelGYM) (Apache-2.0).
Everything listed here was added in this fork by Shrinivas Anavekar; all other
files are upstream. Generated from `git log` / `git diff --stat` against the
upstream snapshot (commit `49e5bd1`).

## Commits

| Date | Commit | Change |
|---|---|---|
| 2026-06-30 | `cc991ee` | add cuda_iterative pipeline, eval scripts, and Trillium build files |
| 2026-06-30 | `90790e8` | finalize SLURM script with Trillium partition/account/GPU directives |
| 2026-06-30 | `0d72c42` | fix SSL_CERT_FILE inside Apptainer container on Trillium |
| 2026-06-30 | `f29db9a` | add .gitattributes to enforce LF line endings for shell/Python/def files |
| 2026-06-30 | `8b1e60e` | fix --gres format: gpu:1 not gpu:h100:1 (type qualifier unsupported on Trillium) |
| 2026-06-30 | `bd42aeb` | switch to compute_full_node partition, drop --gres |
| 2026-06-30 | `6706ea9` | fix Trillium SLURM directives: gpus-per-node, account name, drop mem |
| 2026-07-02 | `cbc3a12` | read HF_TOKEN from file, not env (Trillium injects --export=NONE) |
| 2026-07-06 | `5b79a46` | set TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1 for compute nodes |
| 2026-07-07 | `602b7fd` | fix: give model all max_turns instead of stopping after first failure |
| 2026-07-07 | `322b1c9` | add diagnostic script for raw result error messages |
| 2026-07-07 | `0c02b83` | fix: CUDA_HOME env override + prompt format for ModelNew output |
| 2026-07-07 | `70970ad` | fix: rewrite all 8 reference kernels to use C++ launchers |
| 2026-07-07 | `6fb8caa` | fix: add forward declarations to cpp_sources for all 8 reference kernels |
| 2026-07-10 | `a437f38` | fix: escape \n in cpp_sources string literals for all 8 reference kernels |
| 2026-07-13 | `79cc53c` | feat: roofline-aware terminal reward (novel contribution) |

## Files added or changed

```
 .gitattributes                                     |    6 +
 .gitignore                                         |   10 +-
 Dockerfile.smoke                                   |   27 +
 Dockerfile.trillium                                |   43 +
 drkernel/__init__.py                               |    0
 drkernel/kernel/__init__.py                        |    0
 drkernel/kernel/rewards/__init__.py                |   55 +-
 drkernel/kernel/rewards/iterative_cuda_reward.py   |  618 +++++++++++
 drkernel/kernel/rewards/roofline_reward.py         |  276 +++++
 kernelgym-cuda-trillium.def                        |   51 +
 kernelgym/backend/cuda_iterative.py                |  116 ++
 kernelgym/backend/registry.py                      |    2 +
 kernelgym/schema/__init__.py                       |   10 +
 kernelgym/schema/cuda_iterative_task.py            |  115 ++
 kernelgym/toolkit/cuda_iterative/__init__.py       |    5 +
 kernelgym/toolkit/cuda_iterative/toolkit.py        |  321 ++++++
 kernelgym/toolkit/registry.py                      |    3 +
 kernelgym/workflow/cuda_iterative_optimize.py      |  335 ++++++
 kernelgym/workflow/registry.py                     |    2 +
 .../eval_real/run_drkernel14b_cuda_iterative.py    | 1153 ++++++++++++++++++++
 scripts/eval_real/run_exp1_triton.py               |   92 ++
 scripts/eval_real/start_and_run_experiments.sh     |   95 ++
 scripts/eval_smoke/exp1_triton_smoke.py            |   60 +
 scripts/eval_smoke/exp2_cuda_comparison_smoke.py   |  169 +++
 scripts/eval_smoke/exp2_cuda_new_workflow_smoke.py |  134 +++
 scripts/eval_smoke/exp2_cuda_old_workflow_smoke.py |   62 ++
 scripts/eval_smoke/smoke_cuda_backend.py           |  314 ++++++
 scripts/eval_smoke/smoke_workflow_scripted.py      |  577 ++++++++++
 scripts/run_all_experiments.sh                     |  142 +++
 scripts/run_drkernel14b_trillium.sh                |  100 ++
 scripts/run_evaluation_demonstration.py            |  383 +++++++
 scripts/show_errors.py                             |    9 +
 tests/test_roofline_reward.py                      |  198 ++++
```

## Notes

- New files keep the upstream Apache-2.0 header template, including its
  "Copyright 2025 Bytedance" line; authorship of the files above is recorded in git history.
- The roofline GPU table stores peak values from the project's spec table. The A100 entry's
  dtype label was corrected from fp32 to bf16 (312 TFLOP/s is the BF16/FP16 tensor-core peak).
  Consumer-GPU entries (RTX 3060 Laptop, RTX 5090) should be checked against NVIDIA datasheets
  for dense vs. sparse tensor throughput.
