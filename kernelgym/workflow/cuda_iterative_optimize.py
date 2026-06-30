"""CUDA Iterative Optimization Workflow Controller.

Orchestrates multi-turn kernel optimization with learned STOP criterion.
Main responsibility: orchestrate turn loop, extract CONTINUE/STOP decision, accumulate history.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional
from datetime import datetime

from kernelgym.common import ErrorCode
from kernelgym.config import settings
from kernelgym.core.types import TaskSpec
from kernelgym.core.workflow import WorkflowController, WorkflowState
from kernelgym.core.scheduler import SchedulerAPI
from kernelgym.schema.cuda_iterative_task import (
    CudaIterativeOptimizationTask,
    CudaIterativeEvaluationTask,
    CudaIterativeTrajectoryResult,
)


class CudaIterativeOptimizeWorkflowController(WorkflowController):
    """Orchestrates iterative CUDA kernel optimization with learned stopping."""

    async def validate_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate initial request."""
        try:
            task = CudaIterativeOptimizationTask.from_dict(input_data)
            validation = self._validate_task(task)
            validation["task_id"] = task.task_id
            validation["workflow"] = "cuda_iterative_optimize"
            return validation
        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "task_id": input_data.get("task_id", "unknown"),
                "workflow": "cuda_iterative_optimize",
            }

    async def handle_request(
        self, input_data: Dict[str, Any], scheduler: SchedulerAPI
    ) -> Dict[str, Any]:
        """Main orchestration loop for iterative optimization.

        Supports two modes:
        1. Single-turn (V1): Just evaluate initial kernel
        2. Multi-turn (V2): Handle full optimization trajectory

        Flow:
        1. Parse and validate request
        2. Determine turn number (0 for initial, >0 for subsequent)
        3. Evaluate current kernel (vs turn-0 baseline)
        4. Accumulate turn history
        5. If multi-turn: apply rejection sampling, compute advantages
        6. Return trajectory result with metrics

        Args:
            input_data: Task payload with optional:
                - current_turn: Explicit turn number (for multi-turn calls)
                - current_kernel_code: Updated kernel (for turns > 0)
                - continue_decision: Model's CONTINUE/STOP marker
        """
        try:
            task = CudaIterativeOptimizationTask.from_dict(input_data)
            validation = self._validate_task(task)
            if not validation["valid"]:
                return self._failed_result(
                    task.task_id, validation["errors"][0] if validation["errors"] else "Validation failed"
                )

            # Determine if this is a multi-turn continuation
            current_turn = input_data.get("current_turn", 0)
            is_continuation = current_turn > 0
            continue_marker = input_data.get("continue_decision", "")

            state = WorkflowState(
                {
                    "base_task_id": task.task_id,
                    "turn": current_turn,
                    "trajectory": input_data.get("trajectory", []),  # Load existing history
                    "current_kernel_code": task.initial_cuda_code if current_turn == 0 else input_data.get("current_kernel_code", task.initial_cuda_code),
                    "stop_decision": False,
                    "turn_0_baseline": input_data.get("turn_0_baseline"),
                    "turn_0_test_cases": input_data.get("turn_0_test_cases"),
                }
            )

            # On turn 0: extract baseline and test cases
            if current_turn == 0:
                turn_0_result = await self._evaluate_turn(
                    task, state, scheduler, is_initial_turn=True
                )

                if not turn_0_result["success"]:
                    return self._failed_result(
                        task.task_id, turn_0_result.get("error", "Turn 0 evaluation failed")
                    )

                state.data["turn_0_baseline"] = turn_0_result
                state.data["turn_0_test_cases"] = turn_0_result.get("test_cases")
                state.data["trajectory"].append(turn_0_result)

            else:
                # Turn N > 0: evaluate updated kernel
                eval_result = await self._evaluate_turn(task, state, scheduler, is_initial_turn=False)

                if not eval_result["success"]:
                    return self._failed_result(task.task_id, eval_result.get("error", f"Turn {current_turn} evaluation failed"))

                state.data["trajectory"].append(eval_result)

                # Check stopping criterion
                if self._should_stop(eval_result, continue_marker, current_turn, task.max_turns):
                    state.data["stop_decision"] = True
                    state.data["termination_reason"] = "stop_action"
                elif current_turn >= task.max_turns - 1:
                    state.data["stop_decision"] = True
                    state.data["termination_reason"] = "max_turns"

            return self._build_trajectory_result(task, state)

        except Exception as e:
            import traceback

            traceback.print_exc()
            return self._failed_result(
                input_data.get("task_id", "unknown"),
                f"Workflow error: {str(e)}"
            )

    async def _evaluate_turn(
        self,
        task: CudaIterativeOptimizationTask,
        state: WorkflowState,
        scheduler: SchedulerAPI,
        is_initial_turn: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a single kernel iteration.

        Returns: {success: bool, compiled: bool, correctness: bool, ...}
        """
        current_turn = state.data.get("turn", 0)
        current_kernel = state.data.get("current_kernel_code", task.initial_cuda_code)

        # Create sub-task for this turn's evaluation
        turn_task_id = f"{task.task_id}-turn-{current_turn}"

        eval_task_payload = {
            "task_id": turn_task_id,
            "base_task_id": task.task_id,
            "current_turn": current_turn,
            "current_kernel_code": current_kernel,
            "reference_kernel_code": task.initial_cuda_code,
            "reference_test_cases": state.data.get("turn_0_test_cases"),
            "entry_point": task.entry_point,
            "num_correct_trials": task.num_correct_trials,
            "num_perf_trials": task.num_perf_trials,
            "timeout": task.timeout,
            "device": task.device,
            "enable_profiling": task.enable_profiling,
            "run_correctness": task.run_correctness,
            "run_performance": task.run_performance,
        }

        # Submit evaluation task
        eval_task_spec = TaskSpec(
            kind="cuda_iterative.evaluation",
            payload=eval_task_payload,
            resources=task.resources,
            metadata={"base_task_id": task.task_id, "turn": current_turn},
        )

        try:
            task_id = await scheduler.submit(eval_task_spec)
            result_dict = await scheduler.wait(task_id, timeout=task.timeout)
        except Exception as e:
            return {
                "success": False,
                "error": f"Evaluation task failed: {str(e)}",
            }

        if not result_dict:
            return {
                "success": False,
                "error": "Evaluation returned no result",
            }

        # Extract evaluation metrics
        return {
            "success": True,
            "compiled": result_dict.get("compiled", False),
            "correctness": result_dict.get("correctness", False),
            "kernel_runtime": result_dict.get("kernel_runtime"),
            "reference_runtime": result_dict.get("reference_runtime"),
            "speedup": result_dict.get("speedup"),
            "profiling_data": result_dict.get("profiling_data"),
            "error_message": result_dict.get("error_message"),
            "test_cases": state.data.get("turn_0_test_cases"),  # Pass through
        }

    def _validate_task(self, task: CudaIterativeOptimizationTask) -> Dict[str, Any]:
        """Validate task inputs."""
        errors = []

        if not task.task_id:
            errors.append("task_id is required")
        if not task.initial_cuda_code:
            errors.append("initial_cuda_code is required")
        if task.entry_point not in task.initial_cuda_code:
            errors.append(f"entry_point '{task.entry_point}' not found in initial_cuda_code")
        if "get_inputs" not in task.initial_cuda_code:
            errors.append("initial_cuda_code must define get_inputs() function")
        if task.max_turns < 1:
            errors.append("max_turns must be >= 1")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }

    def _should_stop(
        self,
        eval_result: Dict[str, Any],
        continue_marker: str,
        current_turn: int,
        max_turns: int,
    ) -> bool:
        """Determine if optimization should stop.

        Stops if:
        1. Model explicitly emitted STOP marker
        2. Kernel became incorrect (safeguard)
        3. Reached max_turns (safety cap)

        Args:
            eval_result: Evaluation result from this turn
            continue_marker: Model's decision string
            current_turn: Current turn number
            max_turns: Maximum turns allowed

        Returns:
            True if should stop, False if should continue
        """
        # Check explicit STOP marker
        if "STOP" in continue_marker.upper():
            return True

        # Safeguard: stop if kernel became incorrect
        if not eval_result.get("correctness", False):
            return True

        # Max turns check is done at higher level, but double-check
        if current_turn >= max_turns - 1:
            return True

        return False

    def _failed_result(self, task_id: str, error: str) -> Dict[str, Any]:
        """Build failure result."""
        return {
            "task_id": task_id,
            "status": "failed",
            "error_message": error,
            "trajectory_length": 0,
            "final_kernel_correct": False,
            "termination_reason": "validation_error",
        }

    def _build_trajectory_result(
        self, task: CudaIterativeOptimizationTask, state: WorkflowState
    ) -> Dict[str, Any]:
        """Build final trajectory result from state (V2: multi-turn metrics)."""
        trajectory = state.data.get("trajectory", [])
        turn_0_baseline = state.data.get("turn_0_baseline", {})

        if not trajectory:
            # Only turn 0 evaluated
            trajectory = [turn_0_baseline] if turn_0_baseline else []

        # Extract final kernel correctness
        final_correct = True
        if trajectory:
            final_correct = trajectory[-1].get("correctness", True)

        # Multi-turn metrics
        best_speedup = 1.0
        total_correct_turns = 0
        for turn in trajectory:
            if turn.get("correctness", False):
                speedup = turn.get("speedup", 1.0) or 1.0
                best_speedup = max(best_speedup, speedup)
                total_correct_turns += 1

        # Average speedup across correct turns
        avg_speedup_correct = (
            sum(t.get("speedup", 1.0) or 1.0 for t in trajectory if t.get("correctness", False))
            / max(total_correct_turns, 1)
        )

        # Compute per-turn improvements
        improvements = []
        turn_0_runtime = (turn_0_baseline.get("kernel_runtime") or 0.0) if turn_0_baseline else 0.0
        for turn in trajectory:
            if turn_0_runtime > 0 and turn.get("kernel_runtime"):
                improvement = (turn_0_runtime - turn.get("kernel_runtime")) / turn_0_runtime
            else:
                improvement = 0.0
            improvements.append(improvement)

        result = CudaIterativeTrajectoryResult(
            task_id=task.task_id,
            trajectory_length=len(trajectory),
            final_kernel_correct=final_correct,
            final_kernel_runtime=trajectory[-1].get("kernel_runtime") if trajectory else None,
            best_speedup_across_trajectory=best_speedup,
            termination_reason=state.data.get("termination_reason", "initial_evaluation_only"),
            turn_history=trajectory,
            metadata={
                "avg_speedup_correct_turns": avg_speedup_correct,
                "total_correct_turns": total_correct_turns,
                "total_turns": len(trajectory),
                "per_turn_improvements": improvements,
                "is_stopped": state.data.get("stop_decision", False),
            },
        )

        return result.to_dict()


# Register workflow controller
def register_cuda_iterative_workflow():
    """Register this workflow controller."""
    from kernelgym.workflow.registry import register_workflow

    register_workflow("cuda_iterative_optimize", CudaIterativeOptimizeWorkflowController())
