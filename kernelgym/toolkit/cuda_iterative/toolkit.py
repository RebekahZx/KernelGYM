"""Toolkit for CUDA iterative kernel optimization evaluation.

Evaluates each kernel iteration against turn-0 baseline.
Correctness: does it produce same outputs as turn-0 kernel?
Performance: how much speedup vs turn-0 kernel?
"""

from __future__ import annotations

from typing import Any, Dict
import torch

from kernelgym.common import ErrorCode
from kernelgym.config import settings
from kernelgym.toolkit.base import Toolkit
from kernelgym.toolkit.validation import validate_code
from kernelgym.toolkit.kernelbench.exec_types import set_seed
from kernelgym.toolkit.kernelbench.timing import time_execution_with_cuda_event, get_timing_stats
from kernelgym.toolkit.kernelbench.profiling import compute_triton_kernel_coverage
from kernelgym.toolkit.kernelbench.loading import load_custom_model, graceful_eval_cleanup
from kernelgym.schema.cuda_iterative_task import CudaIterativeEvaluationTask, CudaIterativeResult


class CudaIterativeToolkit(Toolkit):
    """Evaluate CUDA kernels iteratively, comparing against turn-0 baseline."""

    name = "cuda_iterative"

    def evaluate(self, task: Dict[str, Any], backend=None, **kwargs: Any) -> Dict[str, Any]:
        """Main evaluation entry point."""
        eval_task = CudaIterativeEvaluationTask.from_dict(task)
        result = self.evaluate_kernel_iteration(eval_task, backend_adapter=backend)
        return result.to_dict()

    def evaluate_kernel_iteration(
        self, task: CudaIterativeEvaluationTask, backend_adapter=None
    ) -> CudaIterativeResult:
        """
        Evaluate a single kernel iteration.

        Key: correctness is measured against REFERENCE_KERNEL_CODE (turn-0),
        not against PyTorch reference.
        """
        device = torch.device(task.device)

        # Validate reference (turn-0) kernel code
        ref_valid, ref_error = validate_code(task.reference_kernel_code, task.entry_point)
        if not ref_valid:
            return CudaIterativeResult(
                task_id=task.task_id,
                current_turn=task.current_turn,
                compiled=False,
                correctness=False,
                error_message=f"Reference kernel validation failed: {ref_error}",
                metadata={"validation_error": ref_error},
            )

        # Validate current (candidate) kernel code
        kernel_valid, kernel_error = validate_code(task.current_kernel_code, task.entry_point)
        if not kernel_valid:
            return CudaIterativeResult(
                task_id=task.task_id,
                current_turn=task.current_turn,
                compiled=False,
                correctness=False,
                error_message=f"Current kernel validation failed: {kernel_error}",
                metadata={"validation_error": kernel_error},
            )

        # STEP 1: Compile both kernels
        try:
            backend = backend_adapter or self._get_backend("cuda_iterative")

            ref_artifact = backend.compile(
                task.reference_kernel_code,
                device=device,
                entry_point=task.entry_point,
                current_turn=0,
            )
            if not ref_artifact["compiled"]:
                return CudaIterativeResult(
                    task_id=task.task_id,
                    current_turn=task.current_turn,
                    compiled=False,
                    correctness=False,
                    error_message=f"Reference kernel compilation failed: {ref_artifact.get('error')}",
                )

            current_artifact = backend.compile(
                task.current_kernel_code,
                device=device,
                entry_point=task.entry_point,
                current_turn=task.current_turn,
            )
            if not current_artifact["compiled"]:
                return CudaIterativeResult(
                    task_id=task.task_id,
                    current_turn=task.current_turn,
                    compiled=False,
                    correctness=False,
                    error_message=f"Current kernel compilation failed: {current_artifact.get('error')}",
                )
        except Exception as e:
            return CudaIterativeResult(
                task_id=task.task_id,
                current_turn=task.current_turn,
                compiled=False,
                correctness=False,
                error_message=f"Compilation error: {str(e)}",
            )

        # STEP 2: Load both kernels
        try:
            ref_handle = backend.load(ref_artifact, device=device)
            current_handle = backend.load(current_artifact, device=device)
        except Exception as e:
            return CudaIterativeResult(
                task_id=task.task_id,
                current_turn=task.current_turn,
                compiled=False,
                correctness=False,
                error_message=f"Load error: {str(e)}",
            )

        # STEP 3: Extract test functions from reference code context
        ref_context = ref_handle.get("context", {})
        get_inputs_fn = ref_context.get("get_inputs")
        if get_inputs_fn is None:
            backend.cleanup(ref_handle)
            backend.cleanup(current_handle)
            return CudaIterativeResult(
                task_id=task.task_id,
                current_turn=task.current_turn,
                compiled=True,
                correctness=False,
                error_message="Reference kernel code must define get_inputs() function",
            )

        # STEP 4: Create model instances
        try:
            run_correctness = task.run_correctness
            if run_correctness is None:
                run_correctness = True

            torch.cuda.set_device(device)
            set_seed(42)

            correctness_pass = True  # default when correctness check is skipped
            speedup = None
            metadata = {}

            # Create models upfront — needed for either correctness or performance
            try:
                ref_model = backend.create_model(ref_handle, [], device=device)
                current_model = backend.create_model(current_handle, [], device=device)
            except Exception as e:
                backend.cleanup(ref_handle)
                backend.cleanup(current_handle)
                return CudaIterativeResult(
                    task_id=task.task_id,
                    current_turn=task.current_turn,
                    compiled=True,
                    correctness=False,
                    error_message=f"Model creation error: {str(e)}",
                )

            if run_correctness:
                # STEP 5: Correctness check (current vs reference, both CUDA)
                correctness_error = None

                with torch.no_grad():
                    for trial in range(task.num_correct_trials):
                        set_seed(42 + trial)
                        try:
                            inputs = get_inputs_fn()
                            inputs = [
                                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                                for x in inputs
                            ]
                            # Clone inputs so in-place ops in one kernel don't corrupt the other
                            ref_inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
                            cur_inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]

                            set_seed(42 + trial)
                            ref_output = ref_model(*ref_inputs)
                            torch.cuda.synchronize(device=device)

                            set_seed(42 + trial)
                            current_output = current_model(*cur_inputs)
                            torch.cuda.synchronize(device=device)

                            # Check shape match
                            if ref_output.shape != current_output.shape:
                                correctness_pass = False
                                correctness_error = (
                                    f"Trial {trial}: Output shape mismatch. "
                                    f"Expected {ref_output.shape}, got {current_output.shape}"
                                )
                                break

                            # Check numerical match (with tolerance)
                            rtol, atol = 1e-2, 1e-2
                            if not torch.allclose(ref_output, current_output, rtol=rtol, atol=atol):
                                correctness_pass = False
                                correctness_error = (
                                    f"Trial {trial}: Numerical mismatch. "
                                    f"Max diff: {(ref_output - current_output).abs().max().item()}"
                                )
                                break

                        except Exception as e:
                            correctness_pass = False
                            correctness_error = f"Trial {trial}: Execution error: {str(e)}"
                            break

                if not correctness_pass:
                    backend.cleanup(ref_handle)
                    backend.cleanup(current_handle)
                    return CudaIterativeResult(
                        task_id=task.task_id,
                        current_turn=task.current_turn,
                        compiled=True,
                        correctness=False,
                        error_message=correctness_error,
                        metadata=metadata,
                    )

            # STEP 6: Performance measurement
            reference_runtime = None
            current_runtime = None

            run_performance = task.run_performance
            if run_performance is None:
                run_performance = True

            if run_performance and correctness_pass:
                try:
                    with torch.no_grad():
                        # Pre-fetch inputs for each kernel independently
                        ref_inputs_perf = get_inputs_fn()
                        ref_inputs_perf = [
                            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                            for x in ref_inputs_perf
                        ]
                        cur_inputs_perf = get_inputs_fn()
                        cur_inputs_perf = [
                            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                            for x in cur_inputs_perf
                        ]

                        ref_elapsed, _ = time_execution_with_cuda_event(
                            ref_model,
                            *ref_inputs_perf,
                            num_trials=task.num_perf_trials,
                            num_warmup=10,
                            device=device,
                        )
                        reference_runtime = get_timing_stats(ref_elapsed, device=device)["mean"]

                        cur_elapsed, _ = time_execution_with_cuda_event(
                            current_model,
                            *cur_inputs_perf,
                            num_trials=task.num_perf_trials,
                            num_warmup=10,
                            device=device,
                        )
                        current_runtime = get_timing_stats(cur_elapsed, device=device)["mean"]

                    speedup = reference_runtime / current_runtime if current_runtime > 0 else 0.0
                except Exception as e:
                    speedup = 0.0
                    metadata["performance_error"] = str(e)

            # STEP 7: Profiling (if enabled)
            profiling_data = None
            enable_profiling = task.enable_profiling
            if enable_profiling is None:
                enable_profiling = settings.enable_profiling

            if enable_profiling:
                try:
                    profiling_data = compute_triton_kernel_coverage(current_model)
                except Exception as e:
                    metadata["profiling_error"] = str(e)

            # STEP 8: Cleanup
            backend.cleanup(ref_handle)
            backend.cleanup(current_handle)
            graceful_eval_cleanup(ref_context, device)

            return CudaIterativeResult(
                task_id=task.task_id,
                current_turn=task.current_turn,
                compiled=True,
                correctness=correctness_pass if run_correctness else True,
                kernel_runtime=current_runtime,
                reference_runtime=reference_runtime,
                speedup=speedup if (reference_runtime and current_runtime) else None,
                profiling_data=profiling_data,
                metadata=metadata,
            )

        except Exception as e:
            try:
                backend.cleanup(ref_handle)
                backend.cleanup(current_handle)
            except:
                pass
            return CudaIterativeResult(
                task_id=task.task_id,
                current_turn=task.current_turn,
                compiled=True,
                correctness=False,
                error_message=f"Evaluation error: {str(e)}",
            )

    def _get_backend(self, backend_name: str):
        """Get backend by name (adapter pattern)."""
        from kernelgym.backend import get_backend

        return get_backend(backend_name)
