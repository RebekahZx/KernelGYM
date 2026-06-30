#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"

echo "=============================================================================="
echo "Running Dr.Kernel-14B Evaluation: Triton + CUDA Experiments"
echo "=============================================================================="
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check for required commands
check_command() {
  if ! command -v $1 &> /dev/null; then
    echo "[ERROR] Required command not found: $1"
    return 1
  fi
}

echo "[INFO] Checking dependencies..."
check_command python || echo "[WARNING] python3 not in PATH, will try python3"
check_command nvidia-smi || echo "[WARNING] nvidia-smi not found"

echo ""
echo "=============================================================================="
echo "EXPERIMENT 1: Triton Reproduction Baseline"
echo "=============================================================================="
echo ""

cd "$REPO_ROOT"

echo "[INFO] Sourcing grading environment..."
source drkernel/kernel/scripts/eval/grading_common.sh

echo "[INFO] Setting up evaluation parameters..."
FSDP_SIZE=-1
PROJECT_NAME="kernel-grading"
RUN_NAME="drkernel-14b-triton-exp1"
EVAL_DATASET="hkust-nlp/drkernel-validation-data"

MULTI_TURN=False
MAX_USER_TURNS=1

OUTPUT_DIR="$RESULTS_DIR/exp1_triton_repro_baseline"
OUTPUT_PATH="$OUTPUT_DIR/graded_results.parquet"
METRICS_OUTPUT_PATH="$OUTPUT_DIR/metrics.json"
RAW_RESPONSE_PATH="$OUTPUT_DIR/raw_responses.jsonl"

HF_MODEL_PATH="hkust-nlp/drkernel-14b"
MODEL_NAME="${HF_MODEL_PATH}"
MODEL_PATH="${MODEL_NAME}"

N_SAMPLES=1
BATCH_SIZE=8
TEMPERATURE=0.8
TOP_P=0.95
DO_SAMPLE=True

ROLLOUT_MODE="standalone_vllm"
ROLLOUT_GPU_MEMORY_UTIL=0.7
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1

SOLVE_THRESHOLD=0.99
PASS_AT_K=1

REWARD_MANAGER="kernel_async"
REWARD_FUNC_NAME="calculate_reward_speedup"
REWARD_WEIGHTS="0.3_0.4_0.3"

NNODES=1
N_GPUS_PER_NODE=1

echo "[INFO] Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Exp1 Setup Complete"
echo "  Dataset: $EVAL_DATASET"
echo "  Model: $HF_MODEL_PATH"
echo "  Output: $OUTPUT_PATH"
echo "  Backend: triton"
echo "  Multi-turn: False"
echo ""

echo "=============================================================================="
echo "EXPERIMENT 2a: CUDA Old Workflow (kernelbench)"
echo "=============================================================================="
echo ""

RUN_NAME_2A="drkernel-14b-cuda-old-exp2a"
OUTPUT_DIR_2A="$RESULTS_DIR/exp2_cuda_old_workflow"
OUTPUT_PATH_2A="$OUTPUT_DIR_2A/graded_results.parquet"
METRICS_OUTPUT_PATH_2A="$OUTPUT_DIR_2A/metrics.json"
RAW_RESPONSE_PATH_2A="$OUTPUT_DIR_2A/raw_responses.jsonl"

mkdir -p "$OUTPUT_DIR_2A"

echo "[INFO] Exp2a Setup Complete (CUDA with kernelbench backend)"
echo "  Dataset: $EVAL_DATASET (same problems as Exp1, CUDA dataset if available)"
echo "  Model: $HF_MODEL_PATH"
echo "  Output: $OUTPUT_PATH_2A"
echo "  Backend: cuda"
echo "  Workflow: kernelbench"
echo "  Multi-turn: False"
echo ""

echo "=============================================================================="
echo "EXPERIMENT 2b: CUDA New Workflow (iterative)"
echo "=============================================================================="
echo ""

RUN_NAME_2B="drkernel-14b-cuda-new-exp2b"
OUTPUT_DIR_2B="$RESULTS_DIR/exp2_cuda_new_workflow"
OUTPUT_PATH_2B="$OUTPUT_DIR_2B/graded_results.parquet"
METRICS_OUTPUT_PATH_2B="$OUTPUT_DIR_2B/metrics.json"
RAW_RESPONSE_PATH_2B="$OUTPUT_DIR_2B/raw_responses.jsonl"

mkdir -p "$OUTPUT_DIR_2B"

echo "[INFO] Exp2b Setup Complete (CUDA with new iterative workflow)"
echo "  Dataset: $EVAL_DATASET (same problems as Exp2a, identical set)"
echo "  Model: $HF_MODEL_PATH"
echo "  Output: $OUTPUT_PATH_2B"
echo "  Backend: cuda"
echo "  Workflow: cuda_iterative_optimize"
echo "  Max turns: 5"
echo ""

echo "=============================================================================="
echo "READY TO RUN"
echo "=============================================================================="
echo ""
echo "To execute the experiments, run:"
echo "  1. Start KernelGYM server: ./start_all_with_monitor.sh"
echo "  2. Set REWARD_SERVER_URL environment variable with server URL"
echo "  3. Run individual experiment scripts in: scripts/eval_real/"
echo ""
echo "Results will be saved to: $RESULTS_DIR/"
echo ""
