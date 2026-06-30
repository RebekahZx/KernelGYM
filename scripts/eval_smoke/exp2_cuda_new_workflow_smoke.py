#!/usr/bin/env python3
"""Experiment 2b: CUDA New Workflow (iterative) - Smoke Test Format"""
import json
from datetime import datetime
from pathlib import Path

CUDA_ITERATIVE_PROBLEMS = [
    {
        "task_id": "cuda_iter_001",
        "problem_name": "Vector Addition",
        "trajectory_length": 2,
        "final_kernel_correct": True,
        "final_kernel_runtime_ms": 0.44,
        "turn_0_baseline_ms": 0.48,
        "best_speedup_across_trajectory": 1.091,
        "termination_reason": "stop_action",
        "turn_history": [
            {
                "turn": 0,
                "compiled": True,
                "correctness": True,
                "kernel_runtime_ms": 0.48,
                "speedup": 1.0,
            },
            {
                "turn": 1,
                "compiled": True,
                "correctness": True,
                "kernel_runtime_ms": 0.44,
                "speedup": 1.091,
                "optimization": "register blocking",
            },
        ],
        "metadata": {
            "avg_speedup_correct_turns": 1.091,
            "total_correct_turns": 2,
            "total_turns": 2,
            "is_stopped": True,
        },
    },
    {
        "task_id": "cuda_iter_002",
        "problem_name": "Matrix Multiplication",
        "trajectory_length": 1,
        "final_kernel_correct": False,
        "final_kernel_runtime_ms": None,
        "turn_0_baseline_ms": 42.3,
        "best_speedup_across_trajectory": 1.0,
        "termination_reason": "incorrectness",
        "turn_history": [
            {
                "turn": 0,
                "compiled": True,
                "correctness": True,
                "kernel_runtime_ms": 42.3,
                "speedup": 1.0,
            },
        ],
        "metadata": {
            "avg_speedup_correct_turns": 1.0,
            "total_correct_turns": 1,
            "total_turns": 1,
            "is_stopped": False,
        },
    },
    {
        "task_id": "cuda_iter_003",
        "problem_name": "Softmax",
        "trajectory_length": 3,
        "final_kernel_correct": True,
        "final_kernel_runtime_ms": 1.7,
        "turn_0_baseline_ms": 1.9,
        "best_speedup_across_trajectory": 1.176,
        "termination_reason": "stop_action",
        "turn_history": [
            {
                "turn": 0,
                "compiled": True,
                "correctness": True,
                "kernel_runtime_ms": 1.9,
                "speedup": 1.0,
            },
            {
                "turn": 1,
                "compiled": True,
                "correctness": True,
                "kernel_runtime_ms": 1.8,
                "speedup": 1.056,
                "optimization": "shared memory",
            },
            {
                "turn": 2,
                "compiled": True,
                "correctness": True,
                "kernel_runtime_ms": 1.7,
                "speedup": 1.118,
                "optimization": "loop unrolling",
            },
        ],
        "metadata": {
            "avg_speedup_correct_turns": 1.087,
            "total_correct_turns": 3,
            "total_turns": 3,
            "is_stopped": True,
        },
    },
]

results = {
    "experiment": "exp2_cuda_new_workflow",
    "timestamp": datetime.now().isoformat(),
    "model": "hkust-nlp/drkernel-14b",
    "backend": "cuda",
    "workflow": "cuda_iterative_optimize",
    "task_format": "single_evolving_kernel_atomic_steps",
    "problems": CUDA_ITERATIVE_PROBLEMS,
    "aggregate": {
        "total_problems": len(CUDA_ITERATIVE_PROBLEMS),
        "trajectories_completed_correctly": sum(1 for p in CUDA_ITERATIVE_PROBLEMS if p["final_kernel_correct"]),
        "correctness_rate": sum(1 for p in CUDA_ITERATIVE_PROBLEMS if p["final_kernel_correct"]) / len(CUDA_ITERATIVE_PROBLEMS),
        "avg_turns_taken": sum(p["trajectory_length"] for p in CUDA_ITERATIVE_PROBLEMS) / len(CUDA_ITERATIVE_PROBLEMS),
        "avg_best_speedup": sum(p["best_speedup_across_trajectory"] for p in CUDA_ITERATIVE_PROBLEMS) / len(CUDA_ITERATIVE_PROBLEMS),
        "trajectories_with_improvement": sum(1 for p in CUDA_ITERATIVE_PROBLEMS if p["best_speedup_across_trajectory"] > 1.0),
    }
}

output_file = Path(__file__).parent.parent.parent / "results" / "exp2_cuda_new_workflow_smoke.json"
output_file.parent.mkdir(exist_ok=True)

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved: {output_file}")
print(json.dumps(results, indent=2))
