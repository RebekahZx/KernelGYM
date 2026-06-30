#!/usr/bin/env python3
"""
Smoke test: CUDA iterative backend pipeline — no model loading.

Tests the full compilation -> correctness -> speedup -> trajectory-logging
pipeline by feeding hand-written kernel strings directly into
CudaIterativeToolkit. No LLM, no scheduler, no Redis.

Three turns exercised:
  Turn 0  — pure-PyTorch reference (baseline, always correct)
  Turn 1  — custom CUDA vec-add via load_inline (should compile, pass
             correctness, show a measured speedup)
  Turn 2  — wrong kernel (computes a*b not a+b) — should compile but
             FAIL correctness, setting ended_via_stop_but_broken

Verified outputs printed at the end and asserted inline.
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from kernelgym.toolkit.cuda_iterative.toolkit import CudaIterativeToolkit
from kernelgym.schema.cuda_iterative_task import CudaIterativeEvaluationTask
from kernelgym.backend.cuda_iterative import CudaIterativeBackend

# ---------------------------------------------------------------------------
# Kernel strings
# ---------------------------------------------------------------------------

# Turn 0 — pure PyTorch, no custom CUDA extension.
# Used as the reference for correctness in all subsequent turns.
REF_KERNEL = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b

def get_init_inputs():
    return []

def get_inputs():
    n = 1 << 20   # 1M float32 elements
    return [torch.randn(n).cuda(), torch.randn(n).cuda()]
"""

# Turn 1 — hand-optimised CUDA kernel via load_inline.
# Correctness: produces a+b (same as reference). Should show a measured speedup.
TURN1_KERNEL = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_src = r\"\"\"
__global__ void vec_add_f32(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c,
                             int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

torch::Tensor forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.numel() == b.numel(), "input sizes must match");
    auto c = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add_f32<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n
    );
    return c;
}
\"\"\"

_cpp_src = "torch::Tensor forward(torch::Tensor a, torch::Tensor b);"

_ext = load_inline(
    name="smoke_vec_add_v1",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["forward"],
    with_cuda=True,
    extra_cuda_cflags=["-O2", "--use_fast_math"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return _ext.forward(a.contiguous(), b.contiguous())

def get_init_inputs():
    return []

def get_inputs():
    n = 1 << 20
    return [torch.randn(n).cuda(), torch.randn(n).cuda()]
"""

# Turn 2 — intentionally WRONG kernel: computes a*b instead of a+b.
# This should compile fine but FAIL the correctness check, exercising
# the ended_via_stop_but_broken flag.
TURN2_WRONG_KERNEL = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a * b   # intentionally wrong: multiply instead of add

def get_init_inputs():
    return []

def get_inputs():
    n = 1 << 20
    return [torch.randn(n).cuda(), torch.randn(n).cuda()]
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


def make_eval_task(task_id: str, turn: int, current_code: str, ref_code: str,
                   num_perf_trials: int = 20) -> CudaIterativeEvaluationTask:
    return CudaIterativeEvaluationTask(
        task_id=task_id,
        base_task_id="smoke-base",
        current_turn=turn,
        current_kernel_code=current_code,
        reference_kernel_code=ref_code,
        reference_test_cases={},   # toolkit re-execs ref code to get context
        num_correct_trials=3,
        num_perf_trials=num_perf_trials,
        device="cuda:0",
        run_correctness=True,
        run_performance=True,
        enable_profiling=False,    # keep smoke test fast
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    assert torch.cuda.is_available(), "No CUDA device visible — check --gpus flag"
    device_name = torch.cuda.get_device_name(0)
    print(f"\n{'='*60}")
    print(f"  KernelGYM CUDA Backend Smoke Test")
    print(f"  Device: {device_name}")
    print(f"{'='*60}\n")

    toolkit = CudaIterativeToolkit()
    backend = CudaIterativeBackend()

    all_ok = True
    trajectory = []   # accumulate turn results as the workflow would

    # -------------------------------------------------------------------
    # TURN 0 — reference baseline
    # -------------------------------------------------------------------
    print("--- Turn 0: reference baseline (pure PyTorch) ---")
    t0_task = make_eval_task("smoke-t0", 0, REF_KERNEL, REF_KERNEL)
    t0 = toolkit.evaluate_kernel_iteration(t0_task, backend_adapter=backend)
    t0d = t0.to_dict()
    trajectory.append(t0d)

    ok = True
    ok &= check("compiled", t0d["compiled"])
    ok &= check("correctness", t0d["correctness"])
    ok &= check("kernel_runtime is float",
                isinstance(t0d.get("kernel_runtime"), float),
                str(t0d.get("kernel_runtime")))
    ok &= check("no error_message", not t0d.get("error_message"),
                t0d.get("error_message") or "")
    if t0d.get("kernel_runtime"):
        print(f"    baseline runtime: {t0d['kernel_runtime']:.4f} ms")
    all_ok &= ok
    print()

    # -------------------------------------------------------------------
    # TURN 1 — correct custom CUDA kernel (load_inline, nvcc required)
    # -------------------------------------------------------------------
    print("--- Turn 1: correct CUDA kernel via load_inline ---")
    t1_task = make_eval_task("smoke-t1", 1, TURN1_KERNEL, REF_KERNEL)
    t1 = toolkit.evaluate_kernel_iteration(t1_task, backend_adapter=backend)
    t1d = t1.to_dict()
    trajectory.append(t1d)

    ok = True
    ok &= check("compiled", t1d["compiled"], t1d.get("error_message") or "")
    ok &= check("correctness", t1d["correctness"], t1d.get("error_message") or "")
    ok &= check("kernel_runtime is float",
                isinstance(t1d.get("kernel_runtime"), float),
                str(t1d.get("kernel_runtime")))
    ok &= check("speedup is positive",
                isinstance(t1d.get("speedup"), (int, float)) and t1d["speedup"] > 0,
                f"speedup={t1d.get('speedup')}")
    if t1d.get("kernel_runtime") and t1d.get("speedup"):
        print(f"    custom runtime: {t1d['kernel_runtime']:.4f} ms  "
              f"speedup: {t1d['speedup']:.3f}x")
    all_ok &= ok
    print()

    # -------------------------------------------------------------------
    # TURN 2 — wrong kernel: should compile but fail correctness
    # -------------------------------------------------------------------
    print("--- Turn 2: wrong kernel (a*b instead of a+b) ---")
    t2_task = make_eval_task("smoke-t2", 2, TURN2_WRONG_KERNEL, REF_KERNEL)
    t2 = toolkit.evaluate_kernel_iteration(t2_task, backend_adapter=backend)
    t2d = t2.to_dict()
    trajectory.append(t2d)

    ok = True
    ok &= check("compiled", t2d["compiled"])
    ok &= check("correctness is False (expected)", not t2d["correctness"])
    ok &= check("error_message present", bool(t2d.get("error_message")),
                t2d.get("error_message") or "(none)")
    all_ok &= ok
    print()

    # -------------------------------------------------------------------
    # Trajectory-level outcome flags (as eval script would compute them)
    # -------------------------------------------------------------------
    print("--- Trajectory outcome flags ---")

    # final_kernel is turn-2 (last turn attempted)
    final_turn = trajectory[-1]
    final_correct = final_turn.get("correctness", False)

    # ended_via_stop_but_broken: STOP was issued but final kernel is incorrect.
    # Here we simulate the workflow receiving a STOP decision on turn 2.
    simulated_stop_on_turn2 = True
    ended_via_stop_but_broken = simulated_stop_on_turn2 and not final_correct

    # ended_via_max_turns_cap: reached turn cap without STOP — not the case here
    ended_via_max_turns_cap = False

    # ended_via_extraction_failure: extraction failed — not the case here
    ended_via_extraction_failure = False

    # best speedup across the trajectory
    best_speedup = max(
        (t.get("speedup") or 1.0)
        for t in trajectory
        if t.get("correctness", False)
    ) if any(t.get("correctness") for t in trajectory) else None

    ok = True
    ok &= check("ended_via_stop_but_broken == True", ended_via_stop_but_broken)
    ok &= check("ended_via_max_turns_cap == False", not ended_via_max_turns_cap)
    ok &= check("ended_via_extraction_failure == False", not ended_via_extraction_failure)
    ok &= check("best_speedup is float (from turn 1)",
                isinstance(best_speedup, float) and best_speedup > 0,
                f"{best_speedup}")
    all_ok &= ok
    print()

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("--- Per-turn summary ---")
    for i, t in enumerate(trajectory):
        rt = t.get("kernel_runtime")
        sp = t.get("speedup")
        print(f"  Turn {i}: compiled={t['compiled']}  correct={t['correctness']}"
              + (f"  rt={rt:.4f}ms" if rt else "")
              + (f"  speedup={sp:.3f}x" if sp else ""))

    print(f"\n  ended_via_stop_but_broken : {ended_via_stop_but_broken}")
    print(f"  ended_via_max_turns_cap   : {ended_via_max_turns_cap}")
    print(f"  ended_via_extraction_fail : {ended_via_extraction_failure}")
    print(f"  best_speedup_across_traj  : {best_speedup}")

    print(f"\n{'='*60}")
    if all_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — review output above")
    print(f"{'='*60}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
