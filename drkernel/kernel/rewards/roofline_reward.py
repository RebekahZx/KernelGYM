# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Roofline-aware terminal reward for iterative CUDA kernel optimization.

This is the PRIMARY novel contribution (the reward that trains the model past
the degenerate early-stopping baseline). It is intentionally kept COMPLETELY
SEPARATE from ``iterative_cuda_reward.py``: this module has no import-time or
runtime dependency on any baseline reward code, so its logic can be audited in
isolation for the paper.

The roofline model (Williams, Waterman & Patterson, CACM 2009) bounds a kernel's
achievable performance by the hardware. A kernel is classified by its arithmetic
intensity (AI = FLOPs / bytes):

    - AI <  ridge_point  ->  MEMORY-bound  (ceiling = peak_bandwidth * AI)
    - AI >= ridge_point  ->  COMPUTE-bound (ceiling = peak_flops)

where ridge_point = peak_flops / peak_bandwidth. Roofline efficiency is the
fraction of that ceiling a kernel actually reaches; a kernel at 100% efficiency
sits on the roofline and cannot be sped up further without changing the algorithm.

The reward (min-form / PURE-style, terminal only):

    R = C_final * clip(efficiency_final - efficiency_baseline, 0.0, 1.0)

This rewards *progress toward the hardware ceiling* rather than raw speedup, so a
low-intensity kernel cannot game the signal by simply doing less work.

Standalone import (matches the repo convention: ``drkernel/`` is placed on
``sys.path`` at runtime, so the package root is ``kernel``)::

    from kernel.rewards.roofline_reward import roofline_aware_reward
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses
#
# No RooflineSpecs / KernelEvalResult exist anywhere in the repo schema (checked:
# no `class RooflineSpecs`, no `class KernelEvalResult`, no schema module), so we
# define them here. KernelEvalResult mirrors the per-turn fields already produced
# by the eval pipeline (correctness / kernel_runtime_ms / speedup) plus the two
# roofline quantities (flops, bytes_accessed).
# =============================================================================

@dataclass(frozen=True)
class RooflineSpecs:
    """Peak hardware limits for one GPU, in base SI units.

    peak_flops:     peak achievable throughput, FLOP/s (for `flop_dtype`)
    peak_bandwidth: peak DRAM bandwidth, bytes/s
    gpu_name:       torch.cuda device-name string this entry was matched from
    flop_dtype:     dtype the peak_flops figure refers to (e.g. "bf16", "fp32")
    """

    peak_flops: float
    peak_bandwidth: float
    gpu_name: str
    flop_dtype: str

    @property
    def ridge_point(self) -> float:
        # Ridge point = arithmetic intensity where memory- and compute-bound
        # ceilings meet: below it a kernel is memory-bound, at/above compute-bound.
        return self.peak_flops / self.peak_bandwidth


@dataclass
class KernelEvalResult:
    """Evaluation outcome for a single kernel (turn-0 baseline or final).

    correct:        did the kernel pass the correctness check
    runtime_ms:     measured wall-clock runtime of the kernel, milliseconds
    flops:          total floating-point operations for the problem (analytical)
    bytes_accessed: total bytes moved to/from DRAM (theoretical lower bound:
                    input_bytes + output_bytes)
    speedup:        kernel speedup vs. the reference (used only by the fallback)
    problem_id:     identifier, surfaced in the fallback WARNING for tracking
    """

    correct: bool
    runtime_ms: Optional[float] = None
    flops: Optional[float] = None
    bytes_accessed: Optional[float] = None
    speedup: Optional[float] = None
    problem_id: Optional[str] = None


# =============================================================================
# GPU hardware lookup table
#
# Never hardcoded as a default: an unknown GPU raises ValueError rather than
# silently returning wrong numbers. Values are taken from the project spec table
# and stored in base SI units (FLOP/s, bytes/s).
# =============================================================================

_TFLOP = 1.0e12   # 1 TFLOP/s in FLOP/s
_TB = 1.0e12      # 1 TB/s   in bytes/s
_GB = 1.0e9       # 1 GB/s   in bytes/s

# Each entry: (required lowercase substrings that must ALL appear in the device
# name) -> RooflineSpecs. Order matters: more specific names first.
_GPU_ROOFLINE_TABLE = [
    (("a100",),                    RooflineSpecs(312 * _TFLOP, 2039 * _GB, "A100 80GB", "fp32")),
    (("h100",),                    RooflineSpecs(989 * _TFLOP, 3.35 * _TB, "H100 SXM", "bf16")),
    (("rtx", "3060", "laptop"),    RooflineSpecs(101 * _TFLOP, 336 * _GB, "RTX 3060 Laptop", "fp16")),
    (("rtx", "5090"),              RooflineSpecs(838 * _TFLOP, 1.79 * _TB, "RTX 5090", "fp16")),
]


def _match_gpu_name(device_name: str) -> RooflineSpecs:
    """Pure name->specs lookup (no torch), so it is unit-testable without a GPU.

    Raises ValueError if the name matches no table entry — we never guess specs.
    """
    name = device_name.lower()
    for required_substrings, specs in _GPU_ROOFLINE_TABLE:
        # A table entry matches only if EVERY one of its substrings is present.
        if all(token in name for token in required_substrings):
            return specs
    known = ", ".join(specs.gpu_name for _, specs in _GPU_ROOFLINE_TABLE)
    raise ValueError(
        f"Unknown GPU {device_name!r}: no roofline specs available. "
        f"Add it to _GPU_ROOFLINE_TABLE. Known GPUs: {known}."
    )


def get_gpu_roofline_specs(device: int = 0) -> RooflineSpecs:
    """Query device properties + lookup table. Raise ValueError if unknown GPU.

    torch is imported lazily so this module (and the pure-math functions/tests
    below) can be imported on machines without torch installed.
    """
    import torch  # lazy: only the runtime GPU path needs torch

    if not torch.cuda.is_available():
        raise ValueError(
            "No CUDA device available; cannot determine roofline specs. "
            "Roofline reward requires a real GPU."
        )
    # Device name is the lookup key; get_device_properties confirms the device
    # exists and would raise for an out-of-range index.
    torch.cuda.get_device_properties(device)
    device_name = torch.cuda.get_device_name(device)
    return _match_gpu_name(device_name)


# =============================================================================
# Core roofline math
# =============================================================================

def compute_arithmetic_intensity(flops: float, bytes_accessed: float) -> float:
    """FLOPs / bytes_accessed. Both must be > 0."""
    if flops <= 0 or bytes_accessed <= 0:
        raise ValueError(
            f"Arithmetic intensity needs flops > 0 and bytes_accessed > 0; "
            f"got flops={flops}, bytes_accessed={bytes_accessed}."
        )
    # Arithmetic intensity: FLOPs performed per byte of DRAM traffic.
    return flops / bytes_accessed


def compute_roofline_efficiency(
    achieved_flops_per_second: float,
    arithmetic_intensity: float,
    specs: RooflineSpecs,
) -> float:
    """achieved_performance / roofline_ceiling, clipped to [0.0, 1.0]."""
    if achieved_flops_per_second < 0:
        raise ValueError(
            f"achieved_flops_per_second must be >= 0; got {achieved_flops_per_second}."
        )
    if arithmetic_intensity <= 0:
        raise ValueError(
            f"arithmetic_intensity must be > 0; got {arithmetic_intensity}."
        )
    # Roofline ceiling: memory-bound slope (bandwidth * AI) capped by the compute
    # roof (peak_flops); min() picks whichever limit binds at this intensity.
    roofline_ceiling = min(specs.peak_flops, specs.peak_bandwidth * arithmetic_intensity)
    # Efficiency: fraction of the hardware ceiling actually reached.
    efficiency = achieved_flops_per_second / roofline_ceiling
    # Clip to [0,1]: 1.0 means the kernel sits on the roofline (cannot improve
    # without changing the algorithm); measurement noise can push slightly over 1.
    return max(0.0, min(1.0, efficiency))


def _achieved_flops_per_second(result: KernelEvalResult) -> Optional[float]:
    """Achieved throughput in FLOP/s, or None if flops/runtime are unavailable."""
    if result.flops is None or result.runtime_ms is None or result.runtime_ms <= 0:
        return None
    # Achieved performance = total FLOPs / elapsed seconds (runtime_ms -> s).
    return result.flops / (result.runtime_ms / 1000.0)


def _roofline_efficiency_of(result: KernelEvalResult, specs: RooflineSpecs) -> Optional[float]:
    """Roofline efficiency of one kernel result, or None if inputs are missing."""
    achieved = _achieved_flops_per_second(result)
    if achieved is None or result.bytes_accessed is None or result.flops is None:
        return None
    if result.flops <= 0 or result.bytes_accessed <= 0:
        return None
    ai = compute_arithmetic_intensity(result.flops, result.bytes_accessed)
    return compute_roofline_efficiency(achieved, ai, specs)


# =============================================================================
# The reward
# =============================================================================

def roofline_aware_reward(
    baseline_result: KernelEvalResult,
    final_result: KernelEvalResult,
    specs: RooflineSpecs,
) -> float:
    """Terminal roofline-aware reward.

        R = C_final * clip(efficiency_final - efficiency_baseline, 0.0, 1.0)

    where C_final is 1.0 iff the final kernel is correct and efficiency_* are the
    turn-0 and final roofline efficiencies. Rewards progress toward the hardware
    ceiling, so low-intensity kernels cannot game the signal by doing less work.

    Falls back to a correctness-gated speedup improvement when FLOPs/bytes data
    are missing on either kernel. The fallback logs a WARNING that includes the
    problem/kernel ID so we can track how often it fires during training — it
    never fails silently.
    """
    # C_final: correctness gate. An incorrect final kernel earns 0 regardless of
    # any efficiency it appeared to reach.
    c_final = 1.0 if final_result.correct else 0.0
    if c_final == 0.0:
        return 0.0

    eff_baseline = _roofline_efficiency_of(baseline_result, specs)
    eff_final = _roofline_efficiency_of(final_result, specs)

    if eff_baseline is None or eff_final is None:
        # Fallback: FLOPs/bytes missing, so roofline efficiency is undefined.
        problem_id = final_result.problem_id or baseline_result.problem_id or "<unknown>"
        logger.warning(
            "roofline_aware_reward: missing FLOPs/bytes for problem %s "
            "(baseline_eff=%s, final_eff=%s); falling back to correctness-gated "
            "speedup improvement.",
            problem_id, eff_baseline, eff_final,
        )
        # Speedup improvement of final over baseline (baseline defaults to 1.0x),
        # clipped to [0,1]; still gated by C_final (== 1.0 here).
        baseline_speedup = baseline_result.speedup if baseline_result.speedup is not None else 1.0
        final_speedup = final_result.speedup if final_result.speedup is not None else baseline_speedup
        improvement = final_speedup - baseline_speedup
        return c_final * max(0.0, min(1.0, improvement))

    # Min-form reward: progress in roofline efficiency, clipped to [0,1]. A
    # regression (final worse than baseline) clips to 0.
    progress = eff_final - eff_baseline
    return c_final * max(0.0, min(1.0, progress))
