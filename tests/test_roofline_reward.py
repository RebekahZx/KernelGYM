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

"""Unit tests for the roofline-aware reward (novel contribution).

Modules are loaded directly by file path via importlib rather than through the
``kernel.rewards`` package ``__init__``. This (a) proves roofline_reward.py is
importable standalone with no torch installed, and (b) lets us load the baseline
reward module alongside it to check the two do not cross-contaminate.
"""

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

_REWARDS_DIR = Path(__file__).resolve().parent.parent / "drkernel" / "kernel" / "rewards"


def _load(module_name: str, filename: str):
    """Load a single module file directly, bypassing the package __init__."""
    spec = importlib.util.spec_from_file_location(module_name, _REWARDS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: on Python 3.9, `from __future__ import annotations`
    # makes dataclasses resolve field types via sys.modules[cls.__module__].
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


roofline = _load("roofline_reward_under_test", "roofline_reward.py")

RooflineSpecs = roofline.RooflineSpecs
KernelEvalResult = roofline.KernelEvalResult
compute_arithmetic_intensity = roofline.compute_arithmetic_intensity
compute_roofline_efficiency = roofline.compute_roofline_efficiency
roofline_aware_reward = roofline.roofline_aware_reward


# Simple, exactly-computable hardware: ridge point = 100 / 10 = 10 FLOP/byte.
SPECS = RooflineSpecs(peak_flops=100.0, peak_bandwidth=10.0, gpu_name="TEST", flop_dtype="fp32")


# --- arithmetic intensity ----------------------------------------------------

def test_arithmetic_intensity_basic():
    assert compute_arithmetic_intensity(100.0, 50.0) == 2.0


def test_arithmetic_intensity_requires_positive():
    with pytest.raises(ValueError):
        compute_arithmetic_intensity(0.0, 50.0)
    with pytest.raises(ValueError):
        compute_arithmetic_intensity(100.0, 0.0)


# --- memory-bound vs compute-bound efficiency --------------------------------

def test_memory_bound_efficiency():
    # AI = 2 < ridge (10) -> memory-bound. Ceiling = bandwidth*AI = 10*2 = 20.
    ai = compute_arithmetic_intensity(100.0, 50.0)
    assert ai < SPECS.ridge_point
    # achieved 10 of a 20 ceiling -> 0.5 efficiency.
    eff = compute_roofline_efficiency(10.0, ai, SPECS)
    assert eff == pytest.approx(0.5)


def test_compute_bound_efficiency():
    # AI = 20 >= ridge (10) -> compute-bound. Ceiling = peak_flops = 100.
    ai = compute_arithmetic_intensity(200.0, 10.0)
    assert ai >= SPECS.ridge_point
    # achieved 50 of a 100 ceiling -> 0.5 efficiency.
    eff = compute_roofline_efficiency(50.0, ai, SPECS)
    assert eff == pytest.approx(0.5)


def test_perfect_kernel_at_ceiling():
    # Memory-bound ceiling of 20 fully reached -> efficiency exactly 1.0.
    ai = compute_arithmetic_intensity(100.0, 50.0)
    eff = compute_roofline_efficiency(20.0, ai, SPECS)
    assert eff == pytest.approx(1.0)


def test_efficiency_clipped_above_one():
    # Measurement noise can push achieved slightly over the ceiling; clip to 1.0.
    ai = compute_arithmetic_intensity(100.0, 50.0)
    eff = compute_roofline_efficiency(25.0, ai, SPECS)
    assert eff == 1.0


# --- reward: progress toward the ceiling -------------------------------------

def _result(correct, achieved_flops_per_s=None, ai_flops=100.0, ai_bytes=50.0,
            speedup=None, problem_id="p"):
    """Build a KernelEvalResult; runtime is derived from a target FLOP/s."""
    runtime_ms = None
    flops = ai_flops
    if achieved_flops_per_s is not None:
        runtime_ms = (flops / achieved_flops_per_s) * 1000.0
    return KernelEvalResult(
        correct=correct, runtime_ms=runtime_ms, flops=flops,
        bytes_accessed=ai_bytes, speedup=speedup, problem_id=problem_id,
    )


def test_reward_progress_toward_ceiling():
    # baseline eff = 10/20 = 0.5 ; final eff = 15/20 = 0.75 -> reward 0.25.
    baseline = _result(True, achieved_flops_per_s=10.0)
    final = _result(True, achieved_flops_per_s=15.0)
    assert roofline_aware_reward(baseline, final, SPECS) == pytest.approx(0.25)


def test_reward_regression_is_zero():
    # final worse than baseline (eff 0.25 < 0.5) -> negative progress clipped to 0.
    baseline = _result(True, achieved_flops_per_s=10.0)
    final = _result(True, achieved_flops_per_s=5.0)
    assert roofline_aware_reward(baseline, final, SPECS) == 0.0


def test_reward_incorrect_final_is_zero():
    # Correctness gate: incorrect final earns 0 even at a higher efficiency.
    baseline = _result(True, achieved_flops_per_s=10.0)
    final = _result(False, achieved_flops_per_s=20.0)
    assert roofline_aware_reward(baseline, final, SPECS) == 0.0


# --- unknown GPU -------------------------------------------------------------

def test_unknown_gpu_raises():
    with pytest.raises(ValueError, match="Unknown GPU"):
        roofline._match_gpu_name("NVIDIA Made-Up GPU 9999")


def test_known_gpu_lookup():
    # Sanity: the table resolves real device-name strings to specs.
    assert roofline._match_gpu_name("NVIDIA A100-SXM4-80GB").gpu_name == "A100 80GB"
    assert roofline._match_gpu_name("NVIDIA H100 80GB HBM3").gpu_name == "H100 SXM"
    assert roofline._match_gpu_name(
        "NVIDIA GeForce RTX 3060 Laptop GPU").gpu_name == "RTX 3060 Laptop"
    assert roofline._match_gpu_name("NVIDIA GeForce RTX 5090").gpu_name == "RTX 5090"


# --- fallback ----------------------------------------------------------------

def test_missing_flops_triggers_fallback_with_warning(caplog):
    # No flops/bytes -> roofline efficiency undefined -> fallback to speedup.
    baseline = KernelEvalResult(correct=True, speedup=1.0, problem_id="prob-42")
    final = KernelEvalResult(correct=True, speedup=1.5, problem_id="prob-42")
    with caplog.at_level(logging.WARNING):
        reward = roofline_aware_reward(baseline, final, SPECS)
    # Speedup improvement 1.5 - 1.0 = 0.5, correctness-gated -> 0.5.
    assert reward == pytest.approx(0.5)
    # WARNING must fire and name the problem so fallback rate is trackable.
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert "prob-42" in caplog.text


def test_fallback_incorrect_final_still_zero():
    baseline = KernelEvalResult(correct=True, speedup=1.0, problem_id="p")
    final = KernelEvalResult(correct=False, speedup=3.0, problem_id="p")
    # Correctness gate applies before any fallback math.
    assert roofline_aware_reward(baseline, final, SPECS) == 0.0


# --- cross-contamination -----------------------------------------------------

def test_both_reward_files_importable_together():
    # Load the baseline reward module alongside roofline and confirm the two
    # produce independent outputs from the same trajectory data.
    baseline_mod = _load("iterative_cuda_reward_under_test", "iterative_cuda_reward.py")

    traj = [{"final_kernel_correct": True, "turn_history": []}]
    baseline_rewards = baseline_mod.terminal_correctness_only(traj)
    assert baseline_rewards == [1.0]

    # roofline reward on a correct-but-improving kernel is unaffected by the
    # baseline module having just run.
    b = _result(True, achieved_flops_per_s=10.0)
    f = _result(True, achieved_flops_per_s=15.0)
    assert roofline_aware_reward(b, f, SPECS) == pytest.approx(0.25)

    # The two modules are distinct objects with no shared reward symbol.
    assert baseline_mod is not roofline
    assert not hasattr(baseline_mod, "roofline_aware_reward")
    assert not hasattr(roofline, "terminal_correctness_only")
