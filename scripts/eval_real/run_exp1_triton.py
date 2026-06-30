#!/usr/bin/env python
"""Experiment 1: Run Dr.Kernel-14B on Triton backend (Original kernelbench workflow)"""
import json
import sys
import asyncio
from pathlib import Path
from datetime import datetime
import subprocess

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

async def run_exp1_triton():
    """Run Experiment 1: Triton reproduction baseline"""
    print("=" * 80)
    print("Experiment 1: Triton Reproduction Baseline")
    print("=" * 80)
    print(f"Model: hkust-nlp/drkernel-14b")
    print(f"Backend: triton")
    print(f"Workflow: kernelbench (original)")
    print(f"Dataset: hkust-nlp/drkernel-validation-data")
    print()

    # Call the existing grading script
    eval_script = repo_root / "drkernel" / "kernel" / "scripts" / "eval" / "drkernel-14b-maxturns3.sh"

    if not eval_script.exists():
        print(f"ERROR: Evaluation script not found: {eval_script}")
        return None

    results = {
        "experiment": "exp1_triton_repro_baseline",
        "timestamp": datetime.now().isoformat(),
        "model": "hkust-nlp/drkernel-14b",
        "backend": "triton",
        "workflow": "kernelbench",
        "status": "starting",
        "command": str(eval_script),
    }

    print(f"[INFO] Starting evaluation script: {eval_script}")
    print(f"[INFO] This will run the Dr.Kernel-14B model on Triton kernels")
    print()

    try:
        # Run the bash script
        process = await asyncio.create_subprocess_exec(
            "bash", str(eval_script),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        results["status"] = "completed" if process.returncode == 0 else "failed"
        results["return_code"] = process.returncode

        if stdout:
            print("[STDOUT]")
            print(stdout.decode())
            results["stdout_lines"] = len(stdout.decode().split('\n'))

        if stderr:
            print("[STDERR]")
            print(stderr.decode())
            results["stderr_lines"] = len(stderr.decode().split('\n'))

        # Save results
        output_dir = repo_root / "results"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "exp1_triton_repro_baseline.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[RESULTS] Saved to {output_file}")
        return results

    except Exception as e:
        print(f"ERROR: {e}")
        results["status"] = "error"
        results["error"] = str(e)
        return results


if __name__ == "__main__":
    try:
        results = asyncio.run(run_exp1_triton())
        sys.exit(0 if results and results.get("status") == "completed" else 1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Experiment stopped by user")
        sys.exit(130)
