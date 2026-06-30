#!/bin/bash
# SLURM batch script — drkernel-14b full 8-problem eval on Trillium.
#
# BEFORE SUBMITTING — resolve the three FLAGs marked below.
# Submit with:
#   export HF_TOKEN=<your_token>
#   sbatch scripts/run_drkernel14b_trillium.sh

# ---- Job identity ---------------------------------------------------------
#SBATCH --job-name=drkernel14b-eval
#SBATCH --time=02:00:00

# ---- Output ---------------------------------------------------------------
# $HOME is read-only on Trillium compute nodes — logs go to $SCRATCH.
#SBATCH --output=/scratch/anavekar/logs/drkernel14b_%j.out
#SBATCH --error=/scratch/anavekar/logs/drkernel14b_%j.err

# ---- Allocation -----------------------------------------------------------
# FLAG 1: Replace with your actual allocation account.
# On DRAC clusters this is usually "def-<pi_username>" or a sponsored account.
# Check with: sacctmgr show associations user=$USER format=account
#SBATCH --account=YOUR_ALLOCATION_ACCOUNT

# ---- Node request ---------------------------------------------------------
# FLAG 2: Trillium whole-node vs per-GPU scheduling.
# Per SciNet Trillium docs, GPU nodes are scheduled as whole units.
# If that is correct, --nodes=1 gives you the entire node (typically 4x H100).
# Do NOT also add --gres=gpu:N in whole-node mode — that will conflict.
# Verify against: https://docs.scinet.utoronto.ca/index.php/Trillium
# or run "scontrol show partition <name>" and check SelectType/SelectTypeParameters.
#SBATCH --nodes=1

# FLAG 3: Partition name.
# Trillium's GPU partition name is not confirmed here. Replace "gpu" with the
# actual name shown in: sinfo -o "%P %a %l %G %D %N"
#SBATCH --partition=gpu

# ---------------------------------------------------------------------------
set -euo pipefail

SCRATCH=/scratch/anavekar
REPO=$SCRATCH/KernelGYM
SIF=$SCRATCH/kernelgym-cuda-trillium.sif
RESULTS=$REPO/results
HFCACHE=$SCRATCH/hf_cache
LOGS=$SCRATCH/logs

mkdir -p "$RESULTS" "$HFCACHE" "$LOGS"

# HF_TOKEN: must be set in the environment before sbatch.
# Do NOT hardcode here — this file may end up in git.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Set it before sbatch:" >&2
    echo "  export HF_TOKEN=hf_..." >&2
    exit 1
fi

module load StdEnv/2023 gcc/12.3
module load cuda/12.6

echo "========================================================"
echo "  drkernel-14b eval — Trillium SLURM job $SLURM_JOB_ID"
echo "  Node:    $SLURMD_NODENAME"
echo "  SIF:     $SIF"
echo "  Repo:    $REPO"
echo "  Results: $RESULTS"
echo "========================================================"

# Confirm SIF exists — if missing, print build instructions and abort.
if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found." >&2
    echo "Build it first (on a login node or interactive session):" >&2
    echo "  apptainer build $SIF $REPO/kernelgym-cuda-trillium.def" >&2
    exit 1
fi

# --nv         : pass through NVIDIA GPU driver from host (required for CUDA)
# --bind repo  : repo at /workspace so REPO_ROOT resolves correctly and
#                results/ writes to $REPO/results/ on the host
# --bind hfcache: HF model cache on $SCRATCH (persists between jobs, survives
#                 the read-only $HOME restriction on compute nodes)
# --env HF_HOME: point huggingface_hub cache inside the bind-mounted path
# --env HF_TOKEN: forward the token for gated model download
apptainer exec --nv \
    --bind "$REPO":/workspace \
    --bind "$HFCACHE":/hf_cache \
    --env HF_HOME=/hf_cache \
    --env HF_TOKEN="$HF_TOKEN" \
    "$SIF" \
    python3 /workspace/scripts/eval_real/run_drkernel14b_cuda_iterative.py

echo "Job finished at $(date)"
echo "Results in: $RESULTS"
