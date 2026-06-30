#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
LOGS_DIR="$REPO_ROOT/logs"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

echo "=============================================================================="
echo "Dr.Kernel-14B Evaluation: Full Execution"
echo "=============================================================================="
echo "Repository: $REPO_ROOT"
echo "Results: $RESULTS_DIR"
echo "Logs: $LOGS_DIR"
echo ""

# Step 1: Start KernelGYM infrastructure
echo "[STEP 1] Starting KernelGYM infrastructure..."
echo "This will start: Redis, API server, worker monitor, and GPU workers"
echo ""

cd "$REPO_ROOT"

# Start in background
nohup bash start_all_with_monitor.sh --log-dir "$LOGS_DIR" > "$LOGS_DIR/startup.log" 2>&1 &
STARTUP_PID=$!

echo "[PID] KernelGYM startup: $STARTUP_PID"
echo "[INFO] Waiting for infrastructure to initialize (60 seconds)..."

# Wait for infrastructure to start
sleep 60

# Check if startup succeeded
if ! kill -0 $STARTUP_PID 2>/dev/null; then
  echo "[ERROR] Startup process exited. Check logs:"
  echo "  $LOGS_DIR/startup.log"
  exit 1
fi

echo "[OK] Infrastructure started"
echo ""

# Step 2: Detect server URL
echo "[STEP 2] Detecting KernelGYM server URL..."
KERNELGYM_SERVER_URL=$(grep -E "Server.*http" "$LOGS_DIR/startup.log" 2>/dev/null | tail -1 || echo "http://localhost:10907")
echo "[INFO] Server URL: $KERNELGYM_SERVER_URL"
export KERNELGYM_SERVER_URL
export REWARD_SERVER_URL="$KERNELGYM_SERVER_URL"

echo ""

# Step 3: Run experiments
echo "[STEP 3] Running experiments..."
echo ""

# Exp 1: Triton
echo "--- Experiment 1: Triton Reproduction ---"
cd "$REPO_ROOT"

bash drkernel/kernel/scripts/eval/drkernel-14b-maxturns3.sh \
  --output_path "$RESULTS_DIR/exp1_triton_repro_baseline.parquet" \
  --metrics_output_path "$RESULTS_DIR/exp1_triton_repro_baseline_metrics.json" \
  --raw_response_path "$RESULTS_DIR/exp1_triton_repro_baseline_responses.jsonl" \
  2>&1 | tee "$LOGS_DIR/exp1_triton.log"

echo ""
echo "[COMPLETE] Experiment 1 finished"
echo ""

# Exp 2a: CUDA Old Workflow
echo "--- Experiment 2a: CUDA Old Workflow ---"

# (This would require similar script for CUDA backend)
echo "[PENDING] Experiment 2a setup - requires CUDA dataset and kernelbench backend configuration"

# Exp 2b: CUDA New Workflow
echo "--- Experiment 2b: CUDA New Workflow ---"

echo "[PENDING] Experiment 2b setup - will use new cuda_iterative_optimize workflow"

echo ""
echo "=============================================================================="
echo "Evaluation Complete"
echo "=============================================================================="
echo "Results saved to: $RESULTS_DIR"
echo "Logs saved to: $LOGS_DIR"
echo ""

# Clean up
echo "[INFO] Stopping infrastructure..."
bash "$REPO_ROOT/stop_all.sh" 2>/dev/null || true

echo "[DONE]"
