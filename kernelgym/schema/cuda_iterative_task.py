"""Schema for CUDA iterative optimization tasks.

Single CUDA kernel evolves over turns with learned STOP criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, List


@dataclass
class CudaIterativeOptimizationTask:
    """Task for iterative CUDA kernel optimization with learned stopping."""

    task_id: str
    initial_cuda_code: str  # Turn-0 CUDA kernel (must include get_inputs(), get_cases())
    entry_point: str = "ModelNew"  # Class name for the PyTorch wrapper
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    timeout: int = 300
    device: str = "cuda:0"
    priority: str = "normal"
    max_turns: int = 5  # Safety cap on iterations
    enable_profiling: Optional[bool] = None
    run_correctness: Optional[bool] = None
    run_performance: Optional[bool] = None
    resources: Optional[Dict[str, Any]] = None

    # For tracking state across turns (set by workflow, not client)
    current_turn: int = 0
    turn_history: List[Dict[str, Any]] = field(default_factory=list)  # [{kernel_code, metrics, feedback}, ...]
    current_kernel_code: Optional[str] = None  # Latest kernel version
    turn_0_test_cases: Optional[Dict[str, Any]] = None  # Cached from initial code

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CudaIterativeOptimizationTask":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class CudaIterativeEvaluationTask:
    """Sub-task for evaluating a single kernel iteration."""

    task_id: str
    base_task_id: str  # Parent iterative task ID
    current_turn: int
    current_kernel_code: str
    reference_kernel_code: str  # Turn-0 kernel (for correctness reference)
    reference_test_cases: Dict[str, Any]  # test case functions from turn 0
    toolkit: str = "cuda_iterative"
    backend_adapter: str = "cuda_iterative"
    entry_point: str = "ModelNew"
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    timeout: int = 300
    device: str = "cuda:0"
    priority: str = "normal"
    enable_profiling: Optional[bool] = None
    run_correctness: Optional[bool] = None
    run_performance: Optional[bool] = None
    resources: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CudaIterativeEvaluationTask":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class CudaIterativeResult:
    """Result from a single kernel evaluation."""

    task_id: str
    current_turn: int
    compiled: bool
    correctness: bool
    kernel_runtime: Optional[float] = None
    reference_runtime: Optional[float] = None
    speedup: Optional[float] = None
    error_message: Optional[str] = None
    feedback: Optional[str] = None  # Appended to prompt for next turn
    decoy_kernel: bool = False
    profiling_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CudaIterativeTrajectoryResult:
    """Final result after trajectory completes (STOP or max-turns)."""

    task_id: str
    trajectory_length: int  # Number of turns taken
    final_kernel_correct: bool  # Correctness of final kernel
    final_kernel_runtime: Optional[float] = None
    best_speedup_across_trajectory: Optional[float] = None
    termination_reason: str = "unknown"  # "stop_action", "max_turns", "error"
    error_message: Optional[str] = None
    turn_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
