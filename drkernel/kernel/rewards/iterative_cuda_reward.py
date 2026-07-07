"""Reward functions for iterative CUDA kernel optimization.

Two reward functions:
1. terminal_correctness_only (V1): Reward based ONLY on final kernel correctness
2. per_turn_correctness_speedup (V2, stub): Per-turn rewards based on correctness + speedup
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def terminal_correctness_only(
    trajectories: List[Dict[str, Any]],
    **kwargs: Any,
) -> List[float]:
    """
    Terminal correctness only: reward based on final kernel correctness.

    Intentionally simple (V1 baseline). Reward computed ONLY at trajectory end.
    Expected behavior: model learns that safest strategy is immediate STOP (avoid breaking correctness).

    Args:
        trajectories: List of trajectory results from cuda_iterative workflow
                     Each trajectory has: {
                       "trajectory_length": int,
                       "final_kernel_correct": bool,
                       "termination_reason": str,
                       "turn_history": List[Dict] with per-turn results,
                       ...
                     }
        **kwargs: Additional config (not used in V1)

    Returns:
        List of reward scores (one per trajectory)
        - +1.0 if final kernel is correct
        -  0.0 if final kernel is incorrect (pure binary signal, no added risk-aversion penalty)
    """
    rewards = []

    for trajectory in trajectories:
        reward = 1.0 if trajectory.get("final_kernel_correct", False) else 0.0
        rewards.append(reward)

    return rewards


def per_turn_correctness_speedup(
    trajectories: List[Dict[str, Any]],
    speedup_clip: float = 3.0,
    speedup_weight: float = 1.0,
    correctness_penalty: float = -2.0,
    apply_mrs: bool = True,
    apply_prs: bool = True,
    min_coverage_threshold: float = 0.3,
    **kwargs: Any,
) -> List[float]:
    """
    Per-turn rewards based on correctness + speedup (V2).

    Reward computation:
    - For each turn t in trajectory:
      - If correctness(t) == False: r_t = correctness_penalty (-2.0)
      - Else: r_t = 1.0 + min(speedup(t), speedup_clip)

    Post-processing (Rejection Sampling):
    - MRS (Multi-turn Rejection Sampling): Reject entire trajectory if avg speedup too low
    - PRS (Profiling-based Rejection Sampling): Reject if coverage too low

    Advantage estimation:
    - TRLOO (Turn-level Leave-One-Out): Unbiased per-turn advantage

    Args:
        trajectories: List of trajectory dicts with:
            {
                "turn_history": [
                    {
                        "correctness": bool,
                        "kernel_runtime": float,
                        "reference_runtime": float,
                        "speedup": float,
                        "profiling_data": {...}
                    },
                    ...
                ]
            }
        speedup_clip: Max speedup reward (default 3.0x)
        speedup_weight: Weight for speedup component
        correctness_penalty: Penalty for incorrect kernel (default -2.0)
        apply_mrs: Apply Multi-turn Rejection Sampling
        apply_prs: Apply Profiling-based Rejection Sampling
        min_coverage_threshold: Min coverage for PRS (default 0.3)
        **kwargs: Additional config

    Returns:
        List of aggregated reward scores (one per trajectory)
    """
    rewards = []

    for trajectory in trajectories:
        turn_history = trajectory.get("turn_history", [])
        if not turn_history:
            rewards.append(0.0)
            continue

        # Compute per-turn rewards
        per_turn_rewards = []
        for turn in turn_history:
            if not turn.get("correctness", False):
                # Incorrect kernel: penalty
                r_t = correctness_penalty
            else:
                # Correct kernel: base + speedup bonus
                speedup = turn.get("speedup", 1.0) or 1.0
                speedup_bonus = min(speedup, speedup_clip) - 1.0
                r_t = 1.0 + speedup_weight * speedup_bonus

            per_turn_rewards.append(r_t)

        # Aggregate: mean per-turn reward
        trajectory_reward = sum(per_turn_rewards) / len(per_turn_rewards)

        # MRS: Reject if average speedup too low
        if apply_mrs:
            correct_turns = [t for t in turn_history if t.get("correctness", False)]
            avg_speedup = (
                sum(t.get("speedup", 1.0) or 1.0 for t in correct_turns)
                / max(len(correct_turns), 1)
            )
            if avg_speedup < 0.9:  # Threshold: < 0.9x speedup = lazy optimization
                trajectory_reward = correctness_penalty  # Reject

        # PRS: Reject if coverage too low
        if apply_prs:
            final_turn = turn_history[-1] if turn_history else {}
            profiling_data = final_turn.get("profiling_data", {})
            coverage = profiling_data.get("coverage", 0.0) if profiling_data else 0.0
            if final_turn.get("correctness", False) and coverage < min_coverage_threshold:
                trajectory_reward = correctness_penalty * 0.5  # Soft reject

        rewards.append(trajectory_reward)

    return rewards


def compute_trloo_advantages(
    per_turn_rewards: List[float],
    baseline_type: str = "optimal",
) -> List[float]:
    """
    Compute Turn-level Leave-One-Out (TRLOO) advantages.

    Solves multi-turn RL bias: in GRPO, a turn's baseline includes its own outcome,
    creating self-inclusion bias. TRLOO excludes self when computing baseline.

    Formula:
    - advantage_t = r_t - baseline_t
    - baseline_t = mean(r_i for i != t)  # Leave-one-out baseline

    Args:
        per_turn_rewards: List of per-turn rewards [r_0, r_1, ..., r_T]
        baseline_type: "optimal" (leave-one-out) or "mean" (simple mean, biased)

    Returns:
        List of per-turn advantages [a_0, a_1, ..., a_T]
    """
    if len(per_turn_rewards) == 0:
        return []

    if baseline_type == "optimal":
        if len(per_turn_rewards) == 1:
            # Single-turn: zero baseline avoids a zero-advantage dead-end
            return [0.0]
        advantages = []
        for i in range(len(per_turn_rewards)):
            other_rewards = per_turn_rewards[:i] + per_turn_rewards[i+1:]
            baseline = sum(other_rewards) / len(other_rewards)
            advantages.append(per_turn_rewards[i] - baseline)
        return advantages
    else:
        # Simple baseline (biased, for comparison)
        baseline = sum(per_turn_rewards) / len(per_turn_rewards)
        return [r - baseline for r in per_turn_rewards]


def apply_multi_turn_rejection_sampling(
    trajectories: List[Dict[str, Any]],
    min_avg_speedup: float = 0.9,
) -> List[bool]:
    """
    Multi-turn Rejection Sampling (MRS): Filter low-quality trajectories.

    A trajectory is rejected if:
    - Average speedup across all turns < min_avg_speedup
    - Any turn has correctness=False (broken kernel)

    This prevents the model from learning lazy optimizations
    (making tiny changes that don't actually improve kernel).

    Args:
        trajectories: List of trajectory results
        min_avg_speedup: Min average speedup threshold

    Returns:
        List of booleans: True = accept, False = reject
    """
    accept_mask = []

    for trajectory in trajectories:
        turn_history = trajectory.get("turn_history", [])

        # Check for any incorrect turns
        has_incorrect = any(not turn.get("correctness", False) for turn in turn_history)
        if has_incorrect:
            accept_mask.append(False)
            continue

        # Check average speedup
        speedups = [turn.get("speedup", 1.0) or 1.0 for turn in turn_history]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0

        if avg_speedup < min_avg_speedup:
            accept_mask.append(False)
        else:
            accept_mask.append(True)

    return accept_mask


def apply_profiling_based_rejection_sampling(
    trajectories: List[Dict[str, Any]],
    min_coverage: float = 0.3,
) -> List[bool]:
    """
    Profiling-based Rejection Sampling (PRS): Reject if insufficient coverage.

    A trajectory is rejected if the final kernel's profiling data shows
    low coverage (didn't actually optimize the kernel, just changed cosmetics).

    Requires: enable_profiling=True during evaluation.

    Args:
        trajectories: List of trajectory results
        min_coverage: Min coverage threshold (0.0 to 1.0)

    Returns:
        List of booleans: True = accept, False = reject
    """
    accept_mask = []

    for trajectory in trajectories:
        turn_history = trajectory.get("turn_history", [])
        if not turn_history:
            accept_mask.append(False)
            continue

        final_turn = turn_history[-1]
        profiling_data = final_turn.get("profiling_data", {})

        if not profiling_data:
            # No profiling data; assume accept (can't check)
            accept_mask.append(True)
            continue

        coverage = profiling_data.get("coverage", 0.0)
        if coverage >= min_coverage:
            accept_mask.append(True)
        else:
            accept_mask.append(False)

    return accept_mask


def compute_iterative_cuda_reward_batch(
    trajectories: List[Dict[str, Any]],
    reward_fn_name: str = "terminal_correctness_only",
    compute_advantages: bool = False,
    apply_rejection_sampling: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Batch reward computation for iterative CUDA optimization (V2).

    Main entry point. Selects reward function and optionally computes:
    - Per-turn advantages (TRLOO)
    - Rejection masks (MRS, PRS)

    Args:
        trajectories: List of trajectory results
        reward_fn_name: Reward function ("terminal_correctness_only" or "per_turn_correctness_speedup")
        compute_advantages: If True, compute TRLOO advantages per trajectory
        apply_rejection_sampling: If True, compute MRS/PRS masks
        **kwargs: Passed to reward function (speedup_clip, mrs_threshold, etc.)

    Returns:
        Dict with:
        {
            "rewards": List[float],           # Main reward signal
            "advantages": List[List[float]],  # Optional: per-turn advantages
            "mrs_mask": List[bool],           # Optional: MRS rejection mask
            "prs_mask": List[bool],           # Optional: PRS rejection mask
        }

    Raises:
        ValueError: If reward_fn_name is unknown
    """
    reward_fns = {
        "terminal_correctness_only": terminal_correctness_only,
        "per_turn_correctness_speedup": per_turn_correctness_speedup,
    }

    if reward_fn_name not in reward_fns:
        raise ValueError(
            f"Unknown reward function: {reward_fn_name}. "
            f"Available: {list(reward_fns.keys())}"
        )

    reward_fn = reward_fns[reward_fn_name]
    rewards = reward_fn(trajectories, **kwargs)

    result = {"rewards": rewards}

    # Optional: compute per-turn advantages (TRLOO)
    if compute_advantages:
        advantages = []
        for i, trajectory in enumerate(trajectories):
            turn_history = trajectory.get("turn_history", [])
            if turn_history and reward_fn_name == "per_turn_correctness_speedup":
                # Mirror the exact formula used in per_turn_correctness_speedup()
                correctness_penalty = kwargs.get("correctness_penalty", -2.0)
                speedup_clip = kwargs.get("speedup_clip", 3.0)
                speedup_weight = kwargs.get("speedup_weight", 1.0)
                per_turn_rewards = []
                for turn in turn_history:
                    if not turn.get("correctness", False):
                        r_t = correctness_penalty
                    else:
                        speedup = turn.get("speedup", 1.0) or 1.0
                        speedup_bonus = min(speedup, speedup_clip) - 1.0
                        r_t = 1.0 + speedup_weight * speedup_bonus
                    per_turn_rewards.append(r_t)

                # Compute TRLOO advantages
                adv = compute_trloo_advantages(per_turn_rewards, baseline_type="optimal")
                advantages.append(adv)
            else:
                advantages.append([])

        result["advantages"] = advantages

    # Optional: compute rejection masks
    if apply_rejection_sampling:
        mrs_mask = apply_multi_turn_rejection_sampling(
            trajectories,
            min_avg_speedup=kwargs.get("mrs_min_speedup", 0.9),
        )
        prs_mask = apply_profiling_based_rejection_sampling(
            trajectories,
            min_coverage=kwargs.get("prs_min_coverage", 0.3),
        )
        result["mrs_mask"] = mrs_mask
        result["prs_mask"] = prs_mask

    return result


# =============================================================================
# Integration with Dr.Kernel training pipeline
# =============================================================================

def extract_stop_decision(response: str) -> bool:
    """
    Extract STOP/CONTINUE decision from model response.

    Design choice: regex pattern for simplicity.
    Model outputs kernel code, optionally ending with "# OPTIMIZATION_STOP" to indicate stop.

    Args:
        response: Model-generated text (kernel code + optional stop marker)

    Returns:
        True if response contains stop marker, False otherwise
    """
    import re
    stop_pattern = r"#\s*OPTIMIZATION_STOP\b"
    return bool(re.search(stop_pattern, response, re.IGNORECASE))


def extract_kernel_code(response: str) -> str:
    """
    Extract kernel code from model response.

    Reuses existing extraction logic (adapted from kernel_reward.py).

    Args:
        response: Model-generated response

    Returns:
        Extracted kernel code
    """
    import re

    # Try patterns for explicit kernel markers
    patterns = [
        r"# Optimized Kernel:\s*\n(.*?)(?=# OPTIMIZATION_STOP|$)",
        r"```cuda\s*\n(.*?)```",
        r"```python\s*\n(.*?)```",
        r"```(?:\w+)?\s*\n(.*?)```",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Fallback: return whole response as kernel code
    return response.strip()


def build_turn_n_prompt_simple(
    initial_kernel: str,
    turn_0_metrics: Dict[str, Any],
    previous_kernels: List[Dict[str, Any]],
    current_turn: int,
) -> str:
    """
    Build prompt for turn N.

    Args:
        initial_kernel: Turn-0 kernel code
        turn_0_metrics: Evaluation metrics from turn 0 (runtime, etc.)
        previous_kernels: List of {kernel_code, metrics, feedback} from turns 1..N-1
        current_turn: Current turn number

    Returns:
        Formatted prompt for model to generate next kernel
    """
    prompt_parts = []

    # Title and context
    prompt_parts.append(f"# CUDA Kernel Optimization - Turn {current_turn}")
    prompt_parts.append("")

    # Initial kernel and baseline metrics
    prompt_parts.append("## Original Kernel (Turn 0)")
    prompt_parts.append("```cuda")
    prompt_parts.append(initial_kernel)
    prompt_parts.append("```")
    prompt_parts.append("")

    prompt_parts.append(f"### Turn 0 Performance")
    prompt_parts.append(f"- Runtime: {turn_0_metrics.get('kernel_runtime', 'N/A')} ms")
    prompt_parts.append("")

    # Optimization history
    if previous_kernels:
        prompt_parts.append("## Optimization History")
        for i, prev in enumerate(previous_kernels, start=1):
            prompt_parts.append(f"### Turn {i} Attempt")
            if prev.get("metrics", {}).get("correctness"):
                status = "✓ Correct"
            else:
                status = "✗ Incorrect"
            speedup = prev.get("metrics", {}).get("speedup")
            if speedup:
                prompt_parts.append(f"- Status: {status}, Speedup: {speedup:.2f}x")
            else:
                prompt_parts.append(f"- Status: {status}")
            if prev.get("feedback"):
                prompt_parts.append(f"- Feedback: {prev['feedback']}")
            prompt_parts.append("")

    # Current best kernel (last successful one)
    if previous_kernels and previous_kernels[-1].get("metrics", {}).get("correctness"):
        prompt_parts.append("## Current Best Kernel")
        prompt_parts.append("```cuda")
        prompt_parts.append(previous_kernels[-1].get("kernel_code", ""))
        prompt_parts.append("```")
        prompt_parts.append("")

    # Instructions for this turn
    prompt_parts.append("## Your Task for This Turn")
    prompt_parts.append("Make exactly ONE atomic optimization to the kernel:")
    prompt_parts.append("- Options: shared memory tiling, loop unrolling, register blocking,")
    prompt_parts.append("  memory coalescing, occupancy tuning, or similar single transformation")
    prompt_parts.append("- Maintain numerical correctness")
    prompt_parts.append("")

    prompt_parts.append("## Decision")
    prompt_parts.append("After your optimized kernel, add one of:")
    prompt_parts.append("- `# OPTIMIZATION_CONTINUE` to attempt another optimization next turn")
    prompt_parts.append("- `# OPTIMIZATION_STOP` to stop optimizing and finalize this kernel")
    prompt_parts.append("")

    prompt_parts.append("## Format")
    prompt_parts.append("```cuda")
    prompt_parts.append("// Your optimized CUDA kernel here")
    prompt_parts.append("// ... (include the full PyTorch wrapper if needed)")
    prompt_parts.append("```")
    prompt_parts.append("")

    prompt_parts.append("# OPTIMIZATION_STOP  (or OPTIMIZATION_CONTINUE)")

    return "\n".join(prompt_parts)


def build_turn_n_prompt_with_history(
    initial_kernel: str,
    turn_0_metrics: Dict[str, Any],
    turn_history: List[Dict[str, Any]],  # Full trajectory from turns 0..N-1
    current_turn: int,
) -> str:
    """
    Build prompt for turn N with FULL trajectory history (V2).

    Includes all previous turns for richer context and credit assignment.
    Larger context window but better learning signal.

    Args:
        initial_kernel: Turn-0 kernel code
        turn_0_metrics: Metrics from turn 0
        turn_history: All turns 0..N-1 with code, metrics, feedback
        current_turn: Current turn number (N)

    Returns:
        Formatted prompt for model
    """
    prompt_parts = []

    prompt_parts.append(f"# CUDA Kernel Optimization - Full Trajectory (Turn {current_turn})")
    prompt_parts.append("")

    # Original kernel — shown as Python (it is a Python module with load_inline)
    prompt_parts.append("## Original Kernel (Turn 0)")
    prompt_parts.append("```python")
    prompt_parts.append(initial_kernel)
    prompt_parts.append("```")
    _rt0 = turn_0_metrics.get('kernel_runtime')
    prompt_parts.append(f"### Metrics: runtime={f'{_rt0:.3f}ms' if _rt0 is not None else 'N/A (compilation failed)'}")
    prompt_parts.append("")

    # Full history with cumulative metrics
    if len(turn_history) > 1:
        prompt_parts.append("## Optimization Trajectory")
        best_speedup = 1.0
        for i, turn in enumerate(turn_history[1:], start=1):  # Skip turn 0
            prompt_parts.append(f"### Turn {i}")
            status = "✓ Correct" if turn.get("correctness", False) else "✗ Incorrect"
            speedup = turn.get("speedup", 1.0) or 1.0
            best_speedup = max(best_speedup, speedup) if turn.get("correctness") else best_speedup

            prompt_parts.append(f"**Status:** {status}")
            if turn.get("kernel_runtime"):
                _rt0_cmp = turn_0_metrics.get('kernel_runtime') or 0.0
                prompt_parts.append(
                    f"**Runtime:** {turn.get('kernel_runtime'):.3f}ms "
                    f"(vs turn 0: {_rt0_cmp:.3f}ms)"
                )
            if speedup != 1.0:
                prompt_parts.append(f"**Speedup:** {speedup:.2f}x")

            err = turn.get("error_message")
            if err:
                prompt_parts.append(f"**Error:** {err[:300]}")

            kernel_code = turn.get("kernel_code", "")
            if kernel_code:
                prompt_parts.append("**Code:**")
                prompt_parts.append("```python")
                prompt_parts.append(kernel_code[:800])
                prompt_parts.append("```")
            prompt_parts.append("")

        prompt_parts.append(f"## Summary")
        prompt_parts.append(f"- Turns attempted: {len(turn_history) - 1}")
        prompt_parts.append(f"- Best speedup achieved: {best_speedup:.2f}x")
        prompt_parts.append("")

    # Task description
    prompt_parts.append("## Your Task (Turn {})".format(current_turn))
    prompt_parts.append("Produce an optimized version of the kernel above.")
    prompt_parts.append("- Make one clear improvement (e.g. shared memory, coalescing, unrolling, float4 loads)")
    prompt_parts.append("- Preserve numerical correctness")
    prompt_parts.append("")

    prompt_parts.append("## REQUIRED OUTPUT FORMAT")
    prompt_parts.append("Output the COMPLETE Python module — imports, CUDA source string, ModelNew class,")
    prompt_parts.append("get_init_inputs(), get_inputs(). Do NOT output raw CUDA alone.")
    prompt_parts.append("")
    prompt_parts.append("```python")
    prompt_parts.append("import torch")
    prompt_parts.append("import torch.nn as nn")
    prompt_parts.append("from torch.utils.cpp_extension import load_inline")
    prompt_parts.append("")
    prompt_parts.append("_src = \"\"\"")
    prompt_parts.append("// ... your optimized CUDA kernel(s) ...")
    prompt_parts.append("\"\"\"")
    prompt_parts.append("")
    prompt_parts.append("class ModelNew(nn.Module):")
    prompt_parts.append("    def __init__(self):")
    prompt_parts.append("        super().__init__()")
    prompt_parts.append("        self._ext = load_inline(name=\"...\", cpp_sources=\"\",")
    prompt_parts.append("                                cuda_sources=_src, functions=[\"...\"],")
    prompt_parts.append("                                verbose=False)")
    prompt_parts.append("    def forward(self, ...):")
    prompt_parts.append("        ...")
    prompt_parts.append("")
    prompt_parts.append("def get_init_inputs(): return []")
    prompt_parts.append("def get_inputs(): ...")
    prompt_parts.append("```")
    prompt_parts.append("")
    prompt_parts.append("Then add your decision on the next line:")
    prompt_parts.append("# OPTIMIZATION_CONTINUE")
    prompt_parts.append("or")
    prompt_parts.append("# OPTIMIZATION_STOP")

    return "\n".join(prompt_parts)
