#!/bin/bash
# SLURM batch script — drkernel-14b full 8-problem eval on Trillium.
#
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
# rrg-mmehr has higher priority than def-mmehr for GPU jobs.
# Switch to def-mmehr if rrg runs low on allocation.
#SBATCH --account=rrg-mmehride

# ---- Node/GPU request -----------------------------------------------------
# Trillium compute partition: per-GPU scheduling (not whole-node).
# drkernel-14b (14B bf16) = ~28GB — fits on a single H100 80GB.
# Use gpu:h100:4 if you want all 4 GPUs on the node (faster for batch
# inference but costs 4x allocation — not needed for greedy decoding).
# Trillium GPU request: use --gpus-per-node (not --gres).
# 14B model in bf16 = ~28GB — fits on a single H100 80GB.
# Memory is automatic on Trillium (186GB per GPU) — do not set --mem.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=compute

# ---------------------------------------------------------------------------
set -euo pipefail

SCRATCH=/scratch/anavekar
REPO=$SCRATCH/KernelGYM
SIF=$SCRATCH/kernelgym-cuda-trillium.sif
RESULTS=$REPO/results
HFCACHE=$SCRATCH/hf_cache
LOGS=$SCRATCH/logs

mkdir -p "$RESULTS" "$HFCACHE" "$LOGS"

# HF_TOKEN: Trillium's SLURM wrapper injects --export=NONE so environment
# variables don't survive into the job. Read from a file instead.
# Create it once on the login node:
#   echo "hf_..." > /scratch/anavekar/.hf_token && chmod 600 /scratch/anavekar/.hf_token
HF_TOKEN=$(cat /scratch/anavekar/.hf_token 2>/dev/null)
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: /scratch/anavekar/.hf_token not found or empty." >&2
    echo "  echo 'hf_...' > /scratch/anavekar/.hf_token && chmod 600 /scratch/anavekar/.hf_token" >&2
    exit 1
fi

module load StdEnv/2023 gcc/12.3
module load cuda/12.6

echo "========================================================"
echo "  drkernel-14b eval — Trillium SLURM job $SLURM_JOB_ID"
echo "  Node:    $SLURMD_NODENAME"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
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
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HUB_OFFLINE=1 \
    "$SIF" \
    python3 /workspace/scripts/eval_real/run_drkernel14b_cuda_iterative.py

echo "Job finished at $(date)"
echo "Results in: $RESULTS"
