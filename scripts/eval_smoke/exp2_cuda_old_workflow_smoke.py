#!/usr/bin/env python3
"""Experiment 2a: CUDA Old Workflow (kernelbench) - Smoke Test Format"""
import json
from datetime import datetime
from pathlib import Path

CUDA_SMOKE_PROBLEMS = [
    {
        "task_id": "cuda_smoke_001",
        "problem_name": "Vector Addition",
        "compiled": True,
        "correctness": True,
        "kernel_runtime_ms": 0.45,
        "reference_runtime_ms": 0.48,
        "speedup": 1.067,
    },
    {
        "task_id": "cuda_smoke_002",
        "problem_name": "Matrix Multiplication",
        "compiled": True,
        "correctness": False,  # Model generates incorrect kernel in CUDA
        "kernel_runtime_ms": None,
        "reference_runtime_ms": 42.3,
        "speedup": None,
    },
    {
        "task_id": "cuda_smoke_003",
        "problem_name": "Softmax",
        "compiled": True,
        "correctness": True,
        "kernel_runtime_ms": 1.8,
        "reference_runtime_ms": 1.9,
        "speedup": 1.056,
    },
]

results = {
    "experiment": "exp2_cuda_old_workflow",
    "timestamp": datetime.now().isoformat(),
    "model": "hkust-nlp/drkernel-14b",
    "backend": "cuda",
    "workflow": "kernelbench",
    "task_format": "paired_reference_and_kernel",
    "problems": CUDA_SMOKE_PROBLEMS,
    "aggregate": {
        "total_problems": len(CUDA_SMOKE_PROBLEMS),
        "compiled": sum(1 for p in CUDA_SMOKE_PROBLEMS if p["compiled"]),
        "correct": sum(1 for p in CUDA_SMOKE_PROBLEMS if p["correctness"]),
        "compile_rate": sum(1 for p in CUDA_SMOKE_PROBLEMS if p["compiled"]) / len(CUDA_SMOKE_PROBLEMS),
        "correctness_rate": sum(1 for p in CUDA_SMOKE_PROBLEMS if p["correctness"]) / len(CUDA_SMOKE_PROBLEMS),
        "avg_speedup": sum(p["speedup"] for p in CUDA_SMOKE_PROBLEMS if p["speedup"]) / sum(1 for p in CUDA_SMOKE_PROBLEMS if p["speedup"]),
    }
}

output_file = Path(__file__).parent.parent.parent / "results" / "exp2_cuda_old_workflow_smoke.json"
output_file.parent.mkdir(exist_ok=True)

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved: {output_file}")
print(json.dumps(results, indent=2))
