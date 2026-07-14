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

# Relative imports so this package resolves under any name (both the runtime
# `kernel.rewards` entry point and the proper `drkernel.kernel.rewards` package
# import) without depending on a prior sys.path.insert.
#
# The batch reward pulls in runtime infra (ray/httpx/verl via reward_client);
# guard it so the pure-stdlib iterative and roofline modules below stay
# importable from any entry point even when that infra is absent.
try:
    from .kernel_reward import compute_kernel_reward_batch
except ImportError:
    compute_kernel_reward_batch = None

from .iterative_cuda_reward import (
    terminal_correctness_only,
    per_turn_correctness_speedup,
    compute_trloo_advantages,
    apply_multi_turn_rejection_sampling,
    apply_profiling_based_rejection_sampling,
    compute_iterative_cuda_reward_batch,
    extract_stop_decision,
    extract_kernel_code,
    build_turn_n_prompt_simple,
    build_turn_n_prompt_with_history,
)
# Roofline-aware reward (novel contribution): standalone module, no runtime
# dependency on iterative_cuda_reward above.
from .roofline_reward import (
    roofline_aware_reward,
    get_gpu_roofline_specs,
    compute_arithmetic_intensity,
    compute_roofline_efficiency,
    RooflineSpecs,
    KernelEvalResult,
)

__all__ = [
    "compute_kernel_reward_batch",
    "terminal_correctness_only",
    "per_turn_correctness_speedup",
    "compute_trloo_advantages",
    "apply_multi_turn_rejection_sampling",
    "apply_profiling_based_rejection_sampling",
    "compute_iterative_cuda_reward_batch",
    "extract_stop_decision",
    "extract_kernel_code",
    "build_turn_n_prompt_simple",
    "build_turn_n_prompt_with_history",
    "roofline_aware_reward",
    "get_gpu_roofline_specs",
    "compute_arithmetic_intensity",
    "compute_roofline_efficiency",
    "RooflineSpecs",
    "KernelEvalResult",
]
