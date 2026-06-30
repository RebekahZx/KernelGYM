#!/usr/bin/env python3
"""
Scripted-response workflow test: CudaIterativeOptimizeWorkflowController.

Tests the orchestration layer — turn loop, CONTINUE/STOP extraction, outcome
flags, prompt construction, rollback pointer — using a ScriptedScheduler that
returns pre-defined per-turn results. No LLM or real CUDA compilation needed.

Five scenarios:
  1. Happy path      — STOP fires at turn 3; turn 1 had the highest speedup,
                        so best_speedup_across_trajectory(1.8x) ≠ final_speedup(1.2x)
  2. Stop-but-broken — STOP fires at turn 2 on a wrong kernel; rollback pointer
                        (last correct turn index) is printed explicitly for sanity check
  3. Max-turns cap   — 3 turns, all correct, all CONTINUE; verify rollback is NOT
                        exercised (no incorrect turn exists → last_correct == final turn)
  4. Extraction fail — turn-1 response has no code fence; ended_via_extraction_failure
                        set, loop exits gracefully without hitting max-turns
  5. Compile failure — turn-1 response HAS a code fence (extraction succeeds) but
                        the extracted code fails to compile; ended_via_compile_failure
                        set — distinct from extraction failure and correctness failure
"""

import asyncio
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load reward/prompt-builder module directly to avoid verl transitive import
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "iterative_cuda_reward",
    REPO_ROOT / "drkernel" / "kernel" / "rewards" / "iterative_cuda_reward.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
extract_stop_decision       = _mod.extract_stop_decision
build_turn_n_prompt_with_history = _mod.build_turn_n_prompt_with_history
terminal_correctness_only   = _mod.terminal_correctness_only

from kernelgym.core.scheduler import SchedulerAPI
from kernelgym.core.types import TaskSpec
from kernelgym.workflow.cuda_iterative_optimize import CudaIterativeOptimizeWorkflowController

# ---------------------------------------------------------------------------
# Minimal kernel stub — satisfies _validate_task (has ModelNew + get_inputs)
# ---------------------------------------------------------------------------
STUB_KERNEL = """\
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
    n = 1 << 20
    return [torch.randn(n).cuda(), torch.randn(n).cuda()]
"""

# ---------------------------------------------------------------------------
# ScriptedScheduler: per-turn pre-defined results, no GPU
# ---------------------------------------------------------------------------
class ScriptedScheduler(SchedulerAPI):
    """Returns scripted evaluation results keyed by turn number. GPU-free."""

    def __init__(self, turn_results: Dict[int, Dict[str, Any]]) -> None:
        self._scripted = turn_results
        self._pending: Dict[str, Any] = {}

    async def submit(self, task: TaskSpec) -> str:
        task_id = f"scripted-{uuid.uuid4().hex[:8]}"
        turn = task.payload.get("current_turn", 0)
        result = self._scripted.get(turn, {
            "compiled": True, "correctness": True,
            "kernel_runtime": 1.0, "speedup": 1.0,
        })
        self._pending[task_id] = result
        return task_id

    async def wait(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._pending.get(task_id, {})

    async def get_status(self, task_id: str) -> Dict[str, Any]:
        return {"status": "completed"}

    async def cancel(self, task_id: str) -> bool:
        return False


# ---------------------------------------------------------------------------
# Kernel extraction (mirrored from eval script)
# ---------------------------------------------------------------------------
def extract_kernel_with_flag(response: str):
    patterns = [
        r"```cuda\s*\n(.*?)```",
        r"```python\s*\n(.*?)```",
        r"```(?:\w+)?\s*\n(.*?)```",
        r"# Optimized Kernel:\s*\n(.*?)(?=# OPTIMIZATION_STOP|# OPTIMIZATION_CONTINUE|$)",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.DOTALL)
        if m:
            return m.group(1).strip(), False
    return response.strip(), True   # no code fence found


# ---------------------------------------------------------------------------
# Scenario runner — mirrors evaluate_problem() from the eval script
# ---------------------------------------------------------------------------
async def run_scenario(
    name: str,
    scripted_backend: Dict[int, Dict],
    scripted_responses: Dict[int, str],
    max_turns: int,
) -> Dict[str, Any]:
    scheduler = ScriptedScheduler(scripted_backend)
    workflow  = CudaIterativeOptimizeWorkflowController()

    # Turn 0 — establish baseline
    turn0_result = await workflow.handle_request({
        "task_id":           f"wf-{name}",
        "initial_cuda_code": STUB_KERNEL,
        "entry_point":       "ModelNew",
        "num_correct_trials": 1,
        "num_perf_trials":    1,
        "timeout":            60,
        "device":             "cuda:0",
        "max_turns":          max_turns,
        "enable_profiling":   False,
        "current_turn":       0,
    }, scheduler)

    accumulated_history: List[Dict] = list(turn0_result.get("turn_history", []))
    turn_0_baseline = accumulated_history[0] if accumulated_history else {}

    ended_via_stop_but_broken    = False
    ended_via_max_turns_cap      = False
    ended_via_extraction_failure = False
    ended_via_compile_failure    = False
    final_result  = turn0_result
    termination_turn = 0
    prompts_built: List[str] = []

    for turn in range(1, max_turns + 1):
        raw_response = scripted_responses.get(
            turn, "```python\npass\n```\n# OPTIMIZATION_CONTINUE"
        )

        # Prompt construction — must not crash
        prompt = build_turn_n_prompt_with_history(
            initial_kernel=STUB_KERNEL,
            turn_0_metrics=turn_0_baseline,
            turn_history=accumulated_history,
            current_turn=turn,
        )
        prompts_built.append(prompt)

        model_said_stop = extract_stop_decision(raw_response)
        kernel_code, extraction_failed = extract_kernel_with_flag(raw_response)
        if extraction_failed:
            ended_via_extraction_failure = True

        continue_decision = (
            "# OPTIMIZATION_STOP" if model_said_stop else "# OPTIMIZATION_CONTINUE"
        )

        turn_n_result = await workflow.handle_request({
            "task_id":              f"wf-{name}",
            "initial_cuda_code":    STUB_KERNEL,
            "entry_point":          "ModelNew",
            "num_correct_trials":    1,
            "num_perf_trials":       1,
            "timeout":               60,
            "device":                "cuda:0",
            "max_turns":             max_turns,
            "enable_profiling":      False,
            "current_turn":          turn,
            "current_kernel_code":   kernel_code,
            "continue_decision":     continue_decision,
            "trajectory":            accumulated_history,
            "turn_0_baseline":       turn_0_baseline,
            "turn_0_test_cases":     None,
        }, scheduler)

        new_history = turn_n_result.get("turn_history", [])
        if new_history:
            accumulated_history = list(new_history)

        latest = accumulated_history[-1] if accumulated_history else {}
        final_result    = turn_n_result
        termination_turn = turn

        if model_said_stop:
            if not latest.get("correctness", False):
                ended_via_stop_but_broken = True
            break

        if not latest.get("correctness", False):
            if not extraction_failed and not latest.get("compiled", True):
                ended_via_compile_failure = True
            break

        if turn >= max_turns:
            ended_via_max_turns_cap = True
            break

    return {
        "trajectory":                   accumulated_history,
        "final_result":                 final_result,
        "turn_0_baseline":              turn_0_baseline,
        "ended_via_stop_but_broken":    ended_via_stop_but_broken,
        "ended_via_max_turns_cap":      ended_via_max_turns_cap,
        "ended_via_extraction_failure": ended_via_extraction_failure,
        "ended_via_compile_failure":    ended_via_compile_failure,
        "termination_turn":             termination_turn,
        "prompts_built":                prompts_built,
    }


# ---------------------------------------------------------------------------
# Check helper
# ---------------------------------------------------------------------------
_PASS = "\033[32mPASS\033[0m"
_FAIL = "\033[31mFAIL\033[0m"


def check(label: str, cond: bool, detail: str = "") -> bool:
    tag    = _PASS if cond else _FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {label}{suffix}")
    return cond


# ---------------------------------------------------------------------------
# Scenario 1: Happy path — STOP at turn 3; turn 1 was the best turn
# ---------------------------------------------------------------------------
async def scenario_1() -> bool:
    print("=== Scenario 1: Happy path (divergent speedups) ===")

    backend = {
        0: {"compiled": True,  "correctness": True,  "kernel_runtime": 1.000, "reference_runtime": 1.0, "speedup": 1.000},
        1: {"compiled": True,  "correctness": True,  "kernel_runtime": 0.556, "reference_runtime": 1.0, "speedup": 1.800},  # best turn
        2: {"compiled": True,  "correctness": True,  "kernel_runtime": 1.111, "reference_runtime": 1.0, "speedup": 0.900},  # regression
        3: {"compiled": True,  "correctness": True,  "kernel_runtime": 0.833, "reference_runtime": 1.0, "speedup": 1.200},  # STOP (not best)
    }
    responses = {
        1: "```python\npass  # kernel-A\n```\n# OPTIMIZATION_CONTINUE",
        2: "```python\npass  # kernel-B\n```\n# OPTIMIZATION_CONTINUE",
        3: "```python\npass  # kernel-C\n```\n# OPTIMIZATION_STOP",
    }

    out = await run_scenario("s1", backend, responses, max_turns=5)
    traj = out["trajectory"]
    fr   = out["final_result"]

    best_speedup = fr.get("best_speedup_across_trajectory")
    final_rt     = fr.get("final_kernel_runtime")
    t0_rt        = out["turn_0_baseline"].get("kernel_runtime", 1.0)
    final_speedup_computed = (t0_rt / final_rt) if (final_rt and final_rt > 0) else None

    print("\n  Per-turn trajectory:")
    for i, t in enumerate(traj):
        print(f"    Turn {i}: correct={t.get('correctness')}  "
              f"rt={t.get('kernel_runtime')}ms  speedup={t.get('speedup')}")

    print(f"\n  best_speedup_across_trajectory : {best_speedup}")
    print(f"  final_kernel_speedup (t0/rt)   : "
          f"{final_speedup_computed:.4f}" if final_speedup_computed is not None else "  final_kernel_speedup: N/A")
    print(f"  termination_reason             : {fr.get('termination_reason')}")
    print(f"  prompts_built (no crash)       : {len(out['prompts_built'])} prompts")
    print()

    ok = True
    ok &= check("final_kernel_correct = True",
                fr.get("final_kernel_correct") == True)
    ok &= check("termination_reason = stop_action",
                fr.get("termination_reason") == "stop_action")
    ok &= check("ended_via_stop_but_broken = False",
                not out["ended_via_stop_but_broken"])
    ok &= check("ended_via_max_turns_cap = False",
                not out["ended_via_max_turns_cap"])
    ok &= check("4 turns in trajectory (0-3)",
                len(traj) == 4, f"got {len(traj)}")
    ok &= check("best_speedup = 1.8 (turn 1 was best)",
                best_speedup is not None and abs(best_speedup - 1.8) < 0.01,
                f"{best_speedup}")
    ok &= check("final_speedup ≈ 1.2 (turn 3, not the best)",
                final_speedup_computed is not None and abs(final_speedup_computed - 1.2) < 0.05,
                f"{final_speedup_computed}")
    ok &= check("best ≠ final (divergence confirmed)",
                best_speedup != final_speedup_computed)
    ok &= check("all 3 prompt builds succeeded (no crash)",
                all(len(p) > 0 for p in out["prompts_built"]))
    print()
    return ok


# ---------------------------------------------------------------------------
# Scenario 2: Stop-but-broken + explicit rollback pointer
# ---------------------------------------------------------------------------
async def scenario_2() -> bool:
    print("=== Scenario 2: Stop-but-broken + rollback pointer ===")

    backend = {
        0: {"compiled": True,  "correctness": True,  "kernel_runtime": 1.0,  "speedup": 1.000},
        1: {"compiled": True,  "correctness": True,  "kernel_runtime": 0.7,  "speedup": 1.429},  # last correct
        2: {"compiled": True,  "correctness": False, "kernel_runtime": None, "speedup": None,
            "error_message": "Trial 0: Numerical mismatch. Max diff: 12.34"},
    }
    responses = {
        1: "```python\npass  # kernel-A\n```\n# OPTIMIZATION_CONTINUE",
        2: "```python\npass  # kernel-B (wrong output)\n```\n# OPTIMIZATION_STOP",
    }

    out = await run_scenario("s2", backend, responses, max_turns=5)
    traj = out["trajectory"]
    fr   = out["final_result"]

    # Rollback pointer: last trajectory index where correctness=True
    last_correct_idx = max(
        (i for i, t in enumerate(traj) if t.get("correctness", False)),
        default=None,
    )
    last_correct_turn = traj[last_correct_idx] if last_correct_idx is not None else {}

    print("\n  Per-turn trajectory:")
    for i, t in enumerate(traj):
        rollback_marker = "  ← rollback target (last correct)" if i == last_correct_idx else ""
        print(f"    Turn {i}: correct={t.get('correctness')}  "
              f"rt={t.get('kernel_runtime')}ms  speedup={t.get('speedup')}  "
              f"err='{t.get('error_message') or ''}'  {rollback_marker}")

    print(f"\n  ended_via_stop_but_broken       : {out['ended_via_stop_but_broken']}")
    print(f"  last_correct_turn_index         : {last_correct_idx}  "
          f"(turn {last_correct_idx} = last correct before wrong-STOP)")
    print(f"  last_correct_kernel speedup     : {last_correct_turn.get('speedup')}")
    print(f"  last_correct_kernel runtime     : {last_correct_turn.get('kernel_runtime')}ms")
    print(f"  best_speedup_across_trajectory  : {fr.get('best_speedup_across_trajectory')}")
    print(f"  final_kernel_correct            : {fr.get('final_kernel_correct')}")
    print()

    ok = True
    ok &= check("ended_via_stop_but_broken = True",
                out["ended_via_stop_but_broken"])
    ok &= check("final_kernel_correct = False",
                fr.get("final_kernel_correct") == False)
    ok &= check("last_correct_turn_index = 1",
                last_correct_idx == 1, f"got {last_correct_idx}")
    ok &= check("rollback target (turn 1) is correct",
                traj[1].get("correctness") == True)
    ok &= check("rollback target speedup = 1.429",
                abs((last_correct_turn.get("speedup") or 0) - 1.429) < 0.01,
                f"{last_correct_turn.get('speedup')}")
    ok &= check("wrong turn (index 2) has correctness=False",
                traj[2].get("correctness") == False)
    ok &= check("best_speedup = 1.429 (correct turns only)",
                fr.get("best_speedup_across_trajectory") is not None
                and abs(fr["best_speedup_across_trajectory"] - 1.429) < 0.01,
                f"{fr.get('best_speedup_across_trajectory')}")
    ok &= check("ended_via_max_turns_cap = False",
                not out["ended_via_max_turns_cap"])
    print()
    return ok


# ---------------------------------------------------------------------------
# Scenario 3: Max-turns cap — all correct, verify rollback NOT exercised
# ---------------------------------------------------------------------------
async def scenario_3() -> bool:
    print("=== Scenario 3: Max-turns cap (rollback NOT exercised) ===")

    backend = {
        0: {"compiled": True, "correctness": True, "kernel_runtime": 1.000, "speedup": 1.000},
        1: {"compiled": True, "correctness": True, "kernel_runtime": 0.909, "speedup": 1.100},
        2: {"compiled": True, "correctness": True, "kernel_runtime": 0.952, "speedup": 1.050},
        3: {"compiled": True, "correctness": True, "kernel_runtime": 0.869, "speedup": 1.150},
    }
    # All CONTINUE — outer loop will hit max_turns cap, no STOP from model
    responses = {
        1: "```python\npass  # kernel-A\n```\n# OPTIMIZATION_CONTINUE",
        2: "```python\npass  # kernel-B\n```\n# OPTIMIZATION_CONTINUE",
        3: "```python\npass  # kernel-C\n```\n# OPTIMIZATION_CONTINUE",
    }

    out = await run_scenario("s3", backend, responses, max_turns=3)
    traj = out["trajectory"]
    fr   = out["final_result"]

    incorrect_indices = [i for i, t in enumerate(traj) if not t.get("correctness", False)]
    last_correct_idx  = max(
        (i for i, t in enumerate(traj) if t.get("correctness", False)),
        default=None,
    )

    print("\n  Per-turn trajectory:")
    for i, t in enumerate(traj):
        print(f"    Turn {i}: correct={t.get('correctness')}  "
              f"rt={t.get('kernel_runtime')}ms  speedup={t.get('speedup')}")

    print(f"\n  ended_via_max_turns_cap         : {out['ended_via_max_turns_cap']}")
    print(f"  ended_via_stop_but_broken       : {out['ended_via_stop_but_broken']}")
    print(f"  incorrect_turn_indices          : {incorrect_indices}  (must be [])")
    print(f"  last_correct_turn_index         : {last_correct_idx}  "
          f"(== final turn; rollback never needed)")
    print(f"  best_speedup_across_trajectory  : {fr.get('best_speedup_across_trajectory')}")
    print()

    ok = True
    ok &= check("ended_via_max_turns_cap = True",
                out["ended_via_max_turns_cap"])
    ok &= check("ended_via_stop_but_broken = False",
                not out["ended_via_stop_but_broken"])
    ok &= check("no incorrect turns — rollback NOT exercised",
                len(incorrect_indices) == 0,
                f"incorrect at: {incorrect_indices}")
    ok &= check("last_correct_turn = final turn (rollback redundant)",
                last_correct_idx == len(traj) - 1,
                f"last_correct={last_correct_idx}, len={len(traj)}")
    ok &= check("all 4 entries in trajectory (turns 0-3)",
                len(traj) == 4, f"got {len(traj)}")
    ok &= check("best_speedup = 1.15 (peak at turn 3)",
                fr.get("best_speedup_across_trajectory") is not None
                and abs(fr["best_speedup_across_trajectory"] - 1.15) < 0.01,
                f"{fr.get('best_speedup_across_trajectory')}")
    ok &= check("final_kernel_correct = True",
                fr.get("final_kernel_correct") == True)
    print()
    return ok


# ---------------------------------------------------------------------------
# Scenario 4: Extraction failure → graceful termination at turn 1
# ---------------------------------------------------------------------------
async def scenario_4() -> bool:
    print("=== Scenario 4: Extraction failure → graceful termination ===")

    backend = {
        0: {"compiled": True,  "correctness": True,  "kernel_runtime": 1.0, "speedup": 1.0},
        1: {"compiled": False, "correctness": False, "kernel_runtime": None, "speedup": None,
            "error_message": "SyntaxError: invalid syntax (garbage from extraction fallback)"},
    }
    # No code fence → extraction_failed=True; no STOP marker either
    responses = {
        1: ("I think this kernel needs more work. "
            "Consider using shared memory for better cache performance. "
            "Maybe try a tiling approach for the inner loop."),
    }

    out = await run_scenario("s4", backend, responses, max_turns=5)
    traj = out["trajectory"]
    fr   = out["final_result"]

    print("\n  Per-turn trajectory:")
    for i, t in enumerate(traj):
        print(f"    Turn {i}: correct={t.get('correctness')}  "
              f"compiled={t.get('compiled')}  "
              f"err='{t.get('error_message') or ''}'")

    print(f"\n  ended_via_extraction_failure    : {out['ended_via_extraction_failure']}")
    print(f"  ended_via_stop_but_broken       : {out['ended_via_stop_but_broken']}")
    print(f"  ended_via_max_turns_cap         : {out['ended_via_max_turns_cap']}")
    print(f"  final_kernel_correct            : {fr.get('final_kernel_correct')}")
    print(f"  termination_turn                : {out['termination_turn']}")
    print()

    ok = True
    ok &= check("ended_via_extraction_failure = True",
                out["ended_via_extraction_failure"])
    ok &= check("ended_via_stop_but_broken = False",
                not out["ended_via_stop_but_broken"])
    ok &= check("ended_via_max_turns_cap = False",
                not out["ended_via_max_turns_cap"])
    ok &= check("final_kernel_correct = False",
                fr.get("final_kernel_correct") == False)
    ok &= check("loop terminated at turn 1 (not max_turns=5)",
                out["termination_turn"] == 1, f"got {out['termination_turn']}")
    ok &= check("turn 1 compiled=False (garbage code rejected by scripted backend)",
                len(traj) > 1 and traj[1].get("compiled") == False,
                f"traj[1].compiled={traj[1].get('compiled') if len(traj) > 1 else 'N/A'}")
    print()
    return ok


# ---------------------------------------------------------------------------
# Scenario 5: Compile failure — extraction OK, but nvcc rejects the code
# ---------------------------------------------------------------------------
async def scenario_5() -> bool:
    print("=== Scenario 5: Compile failure (extraction OK, code rejected) ===")

    backend = {
        0: {"compiled": True,  "correctness": True,  "kernel_runtime": 1.0, "speedup": 1.0},
        # Turn 1: code fence found (extraction_failed=False), but backend says compiled=False.
        # This is what happens when the model writes syntactically broken CUDA.
        1: {"compiled": False, "correctness": False, "kernel_runtime": None, "speedup": None,
            "error_message": "error: expected ';' before '}' token (nvcc compile error)"},
    }
    # Response HAS a code fence — extraction succeeds. Content is broken CUDA.
    responses = {
        1: "```cuda\n__global__ void bad_kernel(float* a int n {  // syntax error\n}\n```\n# OPTIMIZATION_CONTINUE",
    }

    out = await run_scenario("s5", backend, responses, max_turns=5)
    traj = out["trajectory"]
    fr   = out["final_result"]

    print("\n  Per-turn trajectory:")
    for i, t in enumerate(traj):
        print(f"    Turn {i}: correct={t.get('correctness')}  "
              f"compiled={t.get('compiled')}  "
              f"err='{t.get('error_message') or ''}'")

    print(f"\n  ended_via_compile_failure       : {out['ended_via_compile_failure']}")
    print(f"  ended_via_extraction_failure    : {out['ended_via_extraction_failure']}")
    print(f"  ended_via_stop_but_broken       : {out['ended_via_stop_but_broken']}")
    print(f"  ended_via_max_turns_cap         : {out['ended_via_max_turns_cap']}")
    print(f"  final_kernel_correct            : {fr.get('final_kernel_correct')}")
    print(f"  termination_turn                : {out['termination_turn']}")
    print()

    ok = True
    ok &= check("ended_via_compile_failure = True",
                out["ended_via_compile_failure"])
    ok &= check("ended_via_extraction_failure = False  (code fence WAS found)",
                not out["ended_via_extraction_failure"])
    ok &= check("ended_via_stop_but_broken = False",
                not out["ended_via_stop_but_broken"])
    ok &= check("ended_via_max_turns_cap = False",
                not out["ended_via_max_turns_cap"])
    ok &= check("final_kernel_correct = False",
                fr.get("final_kernel_correct") == False)
    ok &= check("loop terminated at turn 1 (not max_turns=5)",
                out["termination_turn"] == 1, f"got {out['termination_turn']}")
    ok &= check("turn 1 compiled=False",
                len(traj) > 1 and traj[1].get("compiled") == False,
                f"traj[1].compiled={traj[1].get('compiled') if len(traj) > 1 else 'N/A'}")
    ok &= check("compile_failure and extraction_failure are mutually exclusive here",
                out["ended_via_compile_failure"] and not out["ended_via_extraction_failure"])
    print()
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    print(f"\n{'='*60}")
    print("  KernelGYM Workflow Orchestration Test (scripted-response)")
    print(f"{'='*60}\n")

    scenario_fns = [scenario_1, scenario_2, scenario_3, scenario_4, scenario_5]
    results = []
    for fn in scenario_fns:
        results.append(await fn())

    all_ok = all(results)
    print(f"{'='*60}")
    if all_ok:
        print("  ALL SCENARIOS PASSED")
    else:
        failed = [i + 1 for i, r in enumerate(results) if not r]
        print(f"  FAILED: Scenario(s) {failed}")
    print(f"{'='*60}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
