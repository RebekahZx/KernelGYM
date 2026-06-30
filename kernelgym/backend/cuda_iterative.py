"""CUDA iterative backend for single-kernel-evolving-over-turns optimization.

Compares current iteration against turn-0 baseline (same kernel at different optimization stages).
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import torch

from kernelgym.toolkit.kernelbench.loading import load_custom_model
from kernelgym.toolkit.kernelbench.compile import build_compile_cache
from kernelgym.toolkit.validation import validate_code
from kernelgym.backend.kernelbench.base import KernelBenchBackendBase


class CudaIterativeBackend(KernelBenchBackendBase):
    """Backend for iteratively optimized CUDA kernels.

    Key difference from kernelbench CUDA backend:
    - Compares current kernel against REFERENCE kernel (both are CUDA)
    - Correctness check: current_kernel outputs == reference_kernel outputs
    - NOT: current_kernel outputs == pytorch_reference outputs

    This allows optimizing a CUDA kernel while preserving correctness.
    """

    name = "cuda_iterative"

    def compile(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        """Compile CUDA kernel code."""
        device = self._normalize_device(kwargs.get("device"))
        entry_point = kwargs.get("entry_point", "ModelNew")
        backend = kwargs.get("backend", "cuda")
        build_dir = kwargs.get("build_dir")
        turn = kwargs.get("current_turn", 0)

        valid, error = validate_code(code, entry_point)
        if not valid:
            return {
                "compiled": False,
                "error": error,
                "device": str(device),
                "entry_point": entry_point,
                "backend": backend,
                "build_dir": build_dir,
                "turn": turn,
            }

        try:
            compile(code, "<string>", "exec")
        except SyntaxError as exc:
            return {
                "compiled": False,
                "error": f"Syntax error in kernel code: {exc}",
                "device": str(device),
                "entry_point": entry_point,
                "backend": backend,
                "build_dir": build_dir,
                "turn": turn,
            }

        if build_dir is None:
            build_dir = tempfile.mkdtemp(prefix="kernelgym_cuda_iterative_")

        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        cache_result = build_compile_cache(code, build_dir, verbose=False)
        artifact = {
            "compiled": cache_result["compiled"],
            "error": cache_result.get("error"),
            "stdout": cache_result.get("stdout"),
            "stderr": cache_result.get("stderr"),
            "device": str(device),
            "entry_point": entry_point,
            "backend": backend,
            "build_dir": build_dir,
            "code": code,
            "turn": turn,
        }
        return artifact

    def load(self, artifact: Dict[str, Any], **kwargs: Any) -> Any:
        """Load compiled CUDA kernel for execution."""
        code = artifact.get("code")
        entry_point = artifact.get("entry_point", "ModelNew")
        build_dir = artifact.get("build_dir")
        backend = artifact.get("backend", "cuda")
        turn = artifact.get("turn", 0)
        context = kwargs.get("context") or {}

        if not code:
            raise ValueError("CudaIterativeBackend.load requires kernel code in artifact")

        device = self._normalize_device(kwargs.get("device"))
        self._maybe_set_cuda_device(device)

        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        model_cls = load_custom_model(code, context, build_dir)

        if model_cls is None:
            raise ValueError(f"Failed to load model class '{entry_point}' from code")

        return {
            "model_cls": model_cls,
            "tempfile_handle": None,
            "context": context,
            "backend": backend,
            "entry_point": entry_point,
            "device": device,
            "build_dir": build_dir,
            "turn": turn,
            "code": code,
        }
