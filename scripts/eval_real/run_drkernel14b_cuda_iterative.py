#!/usr/bin/env python3
"""
Evaluation: hkust-nlp/drkernel-14b on cuda_iterative_optimize pipeline.

PRE-TRAINING BASELINE. The model was not trained on this iterative CUDA
protocol, so unexpected behavior (never emitting STOP, rewriting full kernel,
emitting Triton instead of CUDA, hitting max-turns cap on most problems) is
expected and should be reported as-is.

Usage:
    # Smoke test: 3 problems, real model, verify output format
    python scripts/eval_real/run_drkernel14b_cuda_iterative.py --smoke

    # Full eval: 8 problems
    python scripts/eval_real/run_drkernel14b_cuda_iterative.py

    # Dry-run: stub model with fixed outputs, verify format without GPU
    python scripts/eval_real/run_drkernel14b_cuda_iterative.py --smoke --dry-run

Outputs:
    results/drkernel14b_new_pipeline_raw.json
    results/drkernel14b_new_pipeline_summary.md
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "drkernel"))  # for kernel.rewards imports

# torchvision is pulled in transitively by transformers' shared image_utils.py
# (modeling_qwen3 -> modeling_layers -> processing_utils -> image_utils).
# It is not used anywhere in the text-only Qwen3-14B inference path — this is
# pure incidental import. If torchvision is absent or its .so was compiled
# against a different torch build (RuntimeError: operator torchvision::nms does
# not exist), stub it before transformers loads so the guard in image_utils.py
# resolves to a no-op mock rather than crashing. Has zero effect when versions
# are matched (the try succeeds and the except block is skipped entirely).
# Torch multimedia extensions (torchvision, torchaudio) are pulled in transitively
# by transformers shared infrastructure (image_utils, loss_rnnt, etc.) and are not
# used anywhere in the text-only Qwen3-14B inference path.
#
# On this machine they are installed but built against a different torch version,
# causing OSError/RuntimeError when the .so/.pyd is loaded. The guard functions
# in transformers (is_torchvision_available, is_torchaudio_available) use
# importlib.util.find_spec which is a filesystem check — it says "available" even
# when the binary is broken, so the guard does not protect us.
#
# Fix: pre-populate sys.modules with proper stubs (types.ModuleType + real
# ModuleSpec) before importing transformers. MagicMock() alone doesn't work here
# because find_spec reads module.__spec__ and raises ValueError on a bare Mock.
# When these packages work correctly (matched torch version), the try blocks
# succeed and the except blocks are skipped — zero effect on the real environment.
def _make_stub(name: str):
    import types as _t
    import importlib.machinery as _im
    m = _t.ModuleType(name)
    m.__spec__ = _im.ModuleSpec(name, loader=None)
    m.__path__ = []
    m.__package__ = name.partition(".")[0]
    return m

try:
    import torchvision  # noqa: F401
except (ImportError, RuntimeError, OSError):
    from unittest.mock import MagicMock as _MM
    _tv_io = _make_stub("torchvision.io")
    _tv_io.ImageReadMode = _MM(name="ImageReadMode")
    _tv_io.decode_image = _MM(name="decode_image")
    _tv_tr = _make_stub("torchvision.transforms")
    _tv_tr.InterpolationMode = _MM(name="InterpolationMode")  # used as dict keys at import time
    _tv_trf = _make_stub("torchvision.transforms.functional")
    _tv_trf.pil_to_tensor = _MM(name="pil_to_tensor")
    _tv = _make_stub("torchvision")
    _tv.io = _tv_io
    _tv.transforms = _tv_tr
    sys.modules.update({
        "torchvision": _tv,
        "torchvision.io": _tv_io,
        "torchvision.transforms": _tv_tr,
        "torchvision.transforms.functional": _tv_trf,
    })

try:
    import torchaudio  # noqa: F401
except (ImportError, RuntimeError, OSError):
    _ta_fn = _make_stub("torchaudio.functional")
    _ta = _make_stub("torchaudio")
    _ta.functional = _ta_fn
    sys.modules.update({
        "torchaudio": _ta,
        "torchaudio.functional": _ta_fn,
    })

from kernelgym.core.scheduler import SchedulerAPI
from kernelgym.core.types import TaskSpec
from kernelgym.workflow.cuda_iterative_optimize import CudaIterativeOptimizeWorkflowController

# Import directly from the module file to avoid the package __init__.py
# which pulls in verl (not required for evaluation).
import importlib.util as _ilu
_reward_spec = _ilu.spec_from_file_location(
    "iterative_cuda_reward",
    REPO_ROOT / "drkernel" / "kernel" / "rewards" / "iterative_cuda_reward.py",
)
_reward_mod = _ilu.module_from_spec(_reward_spec)
_reward_spec.loader.exec_module(_reward_mod)
terminal_correctness_only   = _reward_mod.terminal_correctness_only
extract_stop_decision        = _reward_mod.extract_stop_decision
extract_kernel_code          = _reward_mod.extract_kernel_code
build_turn_n_prompt_with_history = _reward_mod.build_turn_n_prompt_with_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval.drkernel14b")

MODEL_ID = "hkust-nlp/drkernel-14b"
MAX_TURNS = 5
NUM_CORRECT_TRIALS = 5
NUM_PERF_TRIALS = 50  # reduced for eval speed
TIMEOUT = 300
DEVICE = "cuda:0"


# ============================================================
# LOCAL IN-PROCESS SCHEDULER
# Calls the toolkit directly — no Redis or distributed workers.
# ============================================================

class LocalDirectScheduler(SchedulerAPI):
    """
    Runs cuda_iterative.evaluation tasks in-process.
    Suitable for standalone evaluation scripts that don't need the full server.

    dry_run=True: returns plausible mock evaluation results without touching the
    GPU or compiler. Use this to verify output format before a real run.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self._results: Dict[str, Any] = {}
        self._toolkit = None
        self._dry_run = dry_run
        self._turn_counter: Dict[str, int] = {}  # task_id → call count

    def _get_toolkit(self):
        if self._toolkit is None:
            from kernelgym.toolkit.cuda_iterative.toolkit import CudaIterativeToolkit
            self._toolkit = CudaIterativeToolkit()
        return self._toolkit

    async def submit(self, task: TaskSpec) -> str:
        task_id = f"local-{uuid.uuid4().hex[:8]}"
        try:
            result = self._mock_result(task) if self._dry_run else self._run_task(task)
            self._results[task_id] = result
        except Exception as exc:
            logger.error(f"Task {task_id} failed in scheduler: {exc}")
            self._results[task_id] = {
                "compiled": False,
                "correctness": False,
                "error_message": str(exc),
            }
        return task_id

    async def wait(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._results.get(task_id, {})

    async def get_status(self, task_id: str) -> Dict[str, Any]:
        return {"status": "completed" if task_id in self._results else "pending"}

    async def cancel(self, task_id: str) -> bool:
        return False

    def _run_task(self, task: TaskSpec) -> Dict[str, Any]:
        if task.kind == "cuda_iterative.evaluation":
            return self._get_toolkit().evaluate(task.payload)
        raise ValueError(f"Unknown task kind: {task.kind}")

    def _mock_result(self, task: TaskSpec) -> Dict[str, Any]:
        """Return a plausible evaluation result without running anything."""
        p = task.payload
        turn = p.get("current_turn", 0)
        # Simulate: correct kernel, slight speedup that grows with turn
        ref_rt = 1.0  # ms (constant mock baseline)
        cur_rt = 1.0 / (1.0 + turn * 0.05)  # pretend 5% speedup per turn
        return {
            "task_id": p.get("task_id", "mock"),
            "current_turn": turn,
            "compiled": True,
            "correctness": True,
            "kernel_runtime": round(cur_rt, 4),
            "reference_runtime": ref_rt,
            "speedup": round(ref_rt / cur_rt, 4),
            "error_message": None,
            "profiling_data": None,
            "metadata": {"dry_run": True},
        }


# ============================================================
# CUDA PROBLEM SET
# 8 KernelBench-style problems: naive but correct CUDA kernels.
# Each is a self-contained Python module string with:
#   ModelNew, get_init_inputs(), get_inputs(), get_cases()
#
# Problem selection rationale:
#   Covers different optimization opportunities the model might attempt:
#   memory coalescing, block-size tuning, vectorized loads (float4),
#   shared-memory tiling, loop unrolling, occupancy tuning.
# ============================================================

_P01_VEC_ADD = """\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_src = \"\"\"
__global__ void vec_add_k(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
\"\"\"

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._ext = load_inline(name="p01_vec_add", cpp_sources="",
                                cuda_sources=_src, functions=["vec_add_k"],
                                verbose=False)
    def forward(self, a, b):
        out = torch.empty_like(a)
        n = a.numel()
        t = 256
        self._ext.vec_add_k(a, b, out, n, block=(t,), grid=((n + t - 1) // t,))
        return out

def get_init_inputs(): return []
def get_inputs():
    n = 1 << 20
    return [torch.randn(n, device="cuda"), torch.randn(n, device="cuda")]
def get_cases():
    import torch; torch.manual_seed(0)
    a = torch.randn(1 << 20, device="cuda")
    b = torch.randn(1 << 20, device="cuda")
    return [{"inputs": [a, b], "outputs": a + b}]
"""

_P02_RELU = """\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_src = \"\"\"
__global__ void relu_k(const float* __restrict__ x,
                        float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
}
\"\"\"

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._ext = load_inline(name="p02_relu", cpp_sources="",
                                cuda_sources=_src, functions=["relu_k"],
                                verbose=False)
    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        t = 256
        self._ext.relu_k(x, out, n, block=(t,), grid=((n + t - 1) // t,))
        return out

def get_init_inputs(): return []
def get_inputs():
    return [torch.randn(4 << 20, device="cuda")]
def get_cases():
    import torch; torch.manual_seed(1)
    x = torch.randn(4 << 20, device="cuda")
    return [{"inputs": [x], "outputs": torch.relu(x)}]
"""

_P03_ADD_BIAS = """\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_src = \"\"\"
__global__ void add_bias_k(const float* __restrict__ x,
                             const float* __restrict__ b,
                             float* __restrict__ out,
                             int rows, int cols) {
    int r = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (r < rows && c < cols)
        out[r * cols + c] = x[r * cols + c] + b[c];
}
\"\"\"

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._ext = load_inline(name="p03_add_bias", cpp_sources="",
                                cuda_sources=_src, functions=["add_bias_k"],
                                verbose=False)
    def forward(self, x, bias):
        rows, cols = x.shape
        out = torch.empty_like(x)
        t = 256
        cb = (cols + t - 1) // t
        self._ext.add_bias_k(x, bias, out, rows, cols, block=(t,), grid=(rows, cb))
        return out

def get_init_inputs(): return []
def get_inputs():
    return [torch.randn(1024, 1024, device="cuda"),
            torch.randn(1024, device="cuda")]
def get_cases():
    import torch; torch.manual_seed(2)
    x = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    return [{"inputs": [x, b], "outputs": x + b}]
"""

_P04_GELU = """\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_src = \"\"\"
#include <math.h>
__global__ void gelu_k(const float* __restrict__ x,
                        float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float c = 0.7978845608f * (v + 0.044715f * v * v * v);
        y[i] = 0.5f * v * (1.f + tanhf(c));
    }
}
\"\"\"

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._ext = load_inline(name="p04_gelu", cpp_sources="",
                                cuda_sources=_src, functions=["gelu_k"],
                                verbose=False)
    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        t = 256
        self._ext.gelu_k(x, out, n, block=(t,), grid=((n + t - 1) // t,))
        return out

def get_init_inputs(): return []
def get_inputs():
    return [torch.randn(2 << 20, device="cuda")]
def get_cases():
    import torch, torch.nn.functional as F; torch.manual_seed(3)
    x = torch.randn(2 << 20, device="cuda")
    return [{"inputs": [x], "outputs": F.gelu(x, approximate="tanh")}]
"""

_P05_TRANSPOSE = """\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_src = \"\"\"
__global__ void transpose_k(const float* __restrict__ in,
                              float* __restrict__ out,
                              int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols)
        out[c * rows + r] = in[r * cols + c];
}
\"\"\"

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._ext = load_inline(name="p05_transpose", cpp_sources="",
                                cuda_sources=_src, functions=["transpose_k"],
                                verbose=False)
    def forward(self, x):
        rows, cols = x.shape
        out = torch.empty(cols, rows, device=x.device, dtype=x.dtype)
        t = 16
        self._ext.transpose_k(x, out, rows, cols,
                               block=(t, t), grid=((cols + t - 1) // t,
                                                    (rows + t - 1) // t))
        return out

def get_init_inputs(): return []
def get_inputs():
    return [torch.randn(1024, 1024, device="cuda")]
def get_cases():
    import torch; torch.manual_seed(4)
    x = torch.randn(1024, 1024, device="cuda")
    return [{"inputs": [x], "outputs": x.t().contiguous()}]
"""

_P06_ELWISE_MUL = """\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_src = \"\"\"
__global__ void elwise_mul_k(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c,
                               float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i] * scale;
}
\"\"\"

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._ext = load_inline(name="p06_elwise_mul", cpp_sources="",
                                cuda_sources=_src, functions=["elwise_mul_k"],
                                verbose=False)
    def forward(self, a, b):
        out = torch.empty_like(a)
        n = a.numel()
        t = 256
        self._ext.elwise_mul_k(a, b, out, 0.5, n, block=(t,), grid=((n + t - 1) // t,))
        return out

def get_init_inputs(): return []
def get_inputs():
    n = 2 << 20
    return [torch.randn(n, device="cuda"), torch.randn(n, device="cuda")]
def get_cases():
    import torch; torch.manual_seed(5)
    n = 2 << 20
    a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    return [{"inputs": [a, b], "outputs": a * b * 0.5}]
"""

_P07_SOFTMAX = """\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_src = \"\"\"
#include <float.h>
// One warp per row (blockDim.x == 32). Works for cols up to 32*any_int.
__global__ void softmax_k(const float* __restrict__ x,
                            float* __restrict__ y,
                            int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* rx = x + row * cols;
    float*       ry = y + row * cols;

    float mx = -FLT_MAX;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = rx[c]; if (v > mx) mx = v;
    }
    for (int s = 16; s > 0; s >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, s));
    mx = __shfl_sync(0xffffffff, mx, 0);

    float sm = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = expf(rx[c] - mx); ry[c] = v; sm += v;
    }
    for (int s = 16; s > 0; s >>= 1)
        sm += __shfl_down_sync(0xffffffff, sm, s);
    sm = __shfl_sync(0xffffffff, sm, 0);

    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        ry[c] /= sm;
}
\"\"\"

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._ext = load_inline(name="p07_softmax", cpp_sources="",
                                cuda_sources=_src, functions=["softmax_k"],
                                verbose=False)
    def forward(self, x):
        rows, cols = x.shape
        out = torch.empty_like(x)
        self._ext.softmax_k(x, out, rows, cols, block=(32,), grid=(rows,))
        return out

def get_init_inputs(): return []
def get_inputs():
    return [torch.randn(512, 512, device="cuda")]
def get_cases():
    import torch; torch.manual_seed(6)
    x = torch.randn(512, 512, device="cuda")
    return [{"inputs": [x], "outputs": torch.softmax(x, dim=-1)}]
"""

_P08_LAYER_NORM = """\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_src = \"\"\"
#include <math.h>
// One warp per row (blockDim.x == 32). Works for cols up to 32*any_int.
__global__ void layer_norm_k(const float* __restrict__ x,
                               const float* __restrict__ w,
                               const float* __restrict__ b,
                               float* __restrict__ y,
                               int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* rx = x + row * cols;
    float*       ry = y + row * cols;

    float s = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += rx[c];
    for (int stride = 16; stride > 0; stride >>= 1)
        s += __shfl_down_sync(0xffffffff, s, stride);
    float mean = __shfl_sync(0xffffffff, s, 0) / cols;

    float v = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float d = rx[c] - mean; v += d * d;
    }
    for (int stride = 16; stride > 0; stride >>= 1)
        v += __shfl_down_sync(0xffffffff, v, stride);
    float inv_std = rsqrtf(__shfl_sync(0xffffffff, v, 0) / cols + eps);

    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        ry[c] = (rx[c] - mean) * inv_std * w[c] + b[c];
}
\"\"\"

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._ext = load_inline(name="p08_layer_norm", cpp_sources="",
                                cuda_sources=_src, functions=["layer_norm_k"],
                                verbose=False)
    def forward(self, x, weight, bias):
        rows, cols = x.shape
        out = torch.empty_like(x)
        self._ext.layer_norm_k(x, weight, bias, out, rows, cols, 1e-5,
                                block=(32,), grid=(rows,))
        return out

def get_init_inputs(): return []
def get_inputs():
    rows, cols = 512, 512
    return [torch.randn(rows, cols, device="cuda"),
            torch.ones(cols, device="cuda"),
            torch.zeros(cols, device="cuda")]
def get_cases():
    import torch, torch.nn.functional as F; torch.manual_seed(7)
    rows, cols = 512, 512
    x = torch.randn(rows, cols, device="cuda")
    w = torch.ones(cols, device="cuda")
    b = torch.zeros(cols, device="cuda")
    return [{"inputs": [x, w, b],
             "outputs": F.layer_norm(x, (cols,), w, b)}]
"""

# All 8 problems in order. Smoke uses first 3.
ALL_PROBLEMS = [
    ("p01_vec_add",      "Vector Addition (1M)",         _P01_VEC_ADD),
    ("p02_relu",         "ReLU (4M)",                    _P02_RELU),
    ("p03_add_bias",     "Add Bias (1024×1024)",          _P03_ADD_BIAS),
    ("p04_gelu_tanh",    "GELU Tanh Approx (2M)",        _P04_GELU),
    ("p05_mat_transpose","Matrix Transpose (1024×1024)",  _P05_TRANSPOSE),
    ("p06_elwise_mul",   "Elementwise Multiply+Scale (2M)", _P06_ELWISE_MUL),
    ("p07_softmax_rows", "Row Softmax (512×512)",         _P07_SOFTMAX),
    ("p08_layer_norm",   "Layer Norm (512×512)",          _P08_LAYER_NORM),
]
SMOKE_PROBLEMS = ALL_PROBLEMS[:3]


# ============================================================
# KERNEL EXTRACTION (with extraction-failure flag)
# ============================================================

def extract_kernel_with_flag(response: str) -> Tuple[str, bool]:
    """
    Extract kernel code from model response.
    Returns (code, extraction_failed).
    extraction_failed=True means no code fence was found and we fell back
    to using the entire response as the kernel — likely garbage for the compiler.
    """
    patterns = [
        r"```cuda\s*\n(.*?)```",
        r"```python\s*\n(.*?)```",
        r"```(?:\w+)?\s*\n(.*?)```",
        r"# Optimized Kernel:\s*\n(.*?)(?=# OPTIMIZATION_STOP|# OPTIMIZATION_CONTINUE|$)",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.DOTALL)
        if m:
            return m.group(1).strip(), False
    return response.strip(), True  # fallback — extraction failed


# ============================================================
# MODEL LOADING AND INFERENCE
# ============================================================

def load_model(model_id: str, dry_run: bool):
    """Load tokenizer and model. Returns (tokenizer, model) or (None, None) for dry-run."""
    if dry_run:
        return None, None

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    logger.info(f"Loading model: {model_id} (bfloat16, device_map=auto)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded.")
    return tokenizer, model


def generate_response(tokenizer, model, prompt: str, max_new_tokens: int = 2048) -> str:
    """Run greedy generation. Returns raw model output (prompt stripped)."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — reproducible baseline
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = out_ids[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _dry_run_response(turn: int, max_turns: int, initial_kernel: str) -> str:
    """
    Stub model output for dry-run mode.
    Emits the initial kernel unchanged + CONTINUE until the last turn, then STOP.
    Mimics the "model never optimizes" worst-case baseline.
    """
    decision = "# OPTIMIZATION_STOP" if turn >= max_turns - 1 else "# OPTIMIZATION_CONTINUE"
    return f"```python\n{initial_kernel.strip()}\n```\n{decision}\n"


# ============================================================
# SINGLE-PROBLEM EVALUATION LOOP
# ============================================================

async def evaluate_problem(
    problem_id: str,
    problem_name: str,
    initial_kernel: str,
    workflow: CudaIterativeOptimizeWorkflowController,
    scheduler: LocalDirectScheduler,
    tokenizer,
    model,
    max_turns: int,
    dry_run: bool,
) -> Dict[str, Any]:
    """
    Run the full iterative optimization trajectory for one problem.

    Turn 0: evaluate the initial kernel to get the baseline (no model call).
    Turn N>0: call the model with history, evaluate the generated kernel.
    """
    logger.info(f"=== {problem_id}: {problem_name} ===")
    start_time = time.time()

    # ---- TURN 0: establish baseline ----
    turn0_payload = {
        "task_id": f"{problem_id}-run",
        "initial_cuda_code": initial_kernel,
        "entry_point": "ModelNew",
        "num_correct_trials": NUM_CORRECT_TRIALS,
        "num_perf_trials": NUM_PERF_TRIALS,
        "timeout": TIMEOUT,
        "device": DEVICE,
        "max_turns": max_turns,
        "enable_profiling": False,
        "current_turn": 0,
    }
    turn0_result = await workflow.handle_request(turn0_payload, scheduler)
    logger.info(f"  Turn 0: compiled={turn0_result.get('turn_history', [{}])[0].get('compiled')}  "
                f"correct={turn0_result.get('turn_history', [{}])[0].get('correctness')}  "
                f"runtime={turn0_result.get('turn_history', [{}])[0].get('kernel_runtime')}")

    if not turn0_result.get("turn_history"):
        return _failed_problem_result(problem_id, problem_name, "Turn 0 evaluation returned no history")

    accumulated_history = list(turn0_result["turn_history"])
    turn_0_baseline = accumulated_history[0]
    turn_0_test_cases = turn0_result.get("turn_history", [{}])[0].get("test_cases")

    # Outcome flags (updated as trajectory progresses)
    ended_via_stop_but_broken   = False
    ended_via_max_turns_cap     = False
    ended_via_extraction_failure = False
    ended_via_compile_failure   = False   # extraction OK but nvcc rejected the code
    termination_turn = 0
    all_model_responses: List[str] = []

    # ---- TURNS 1 .. max_turns ----
    final_trajectory_result = turn0_result
    for turn in range(1, max_turns + 1):
        # Build prompt with full trajectory history
        prompt = build_turn_n_prompt_with_history(
            initial_kernel=initial_kernel,
            turn_0_metrics=turn_0_baseline,
            turn_history=accumulated_history,
            current_turn=turn,
        )

        # Generate model response
        if dry_run:
            raw_response = _dry_run_response(turn, max_turns, initial_kernel)
        else:
            logger.info(f"  Turn {turn}: generating ...")
            raw_response = generate_response(tokenizer, model, prompt)

        all_model_responses.append(raw_response)
        model_said_stop = extract_stop_decision(raw_response)
        kernel_code, extraction_failed = extract_kernel_with_flag(raw_response)
        if extraction_failed:
            ended_via_extraction_failure = True
            logger.warning(f"  Turn {turn}: extraction fallback triggered (no code fence found)")

        continue_decision = "# OPTIMIZATION_STOP" if model_said_stop else "# OPTIMIZATION_CONTINUE"

        # Evaluate the generated kernel
        turn_n_payload = {
            "task_id": f"{problem_id}-run",
            "initial_cuda_code": initial_kernel,
            "entry_point": "ModelNew",
            "num_correct_trials": NUM_CORRECT_TRIALS,
            "num_perf_trials": NUM_PERF_TRIALS,
            "timeout": TIMEOUT,
            "device": DEVICE,
            "max_turns": max_turns,
            "enable_profiling": False,
            "current_turn": turn,
            "current_kernel_code": kernel_code,
            "continue_decision": continue_decision,
            "trajectory": accumulated_history,
            "turn_0_baseline": turn_0_baseline,
            "turn_0_test_cases": turn_0_test_cases,
        }
        turn_n_result = await workflow.handle_request(turn_n_payload, scheduler)

        # The workflow returns the updated trajectory
        new_history = turn_n_result.get("turn_history", [])
        if new_history:
            accumulated_history = list(new_history)

        latest_turn = accumulated_history[-1] if accumulated_history else {}
        logger.info(
            f"  Turn {turn}: compiled={latest_turn.get('compiled')}  "
            f"correct={latest_turn.get('correctness')}  "
            f"speedup={latest_turn.get('speedup')}  "
            f"stop={model_said_stop}"
        )

        final_trajectory_result = turn_n_result
        termination_turn = turn

        # Determine outcome flags
        if model_said_stop:
            if not latest_turn.get("correctness", False):
                ended_via_stop_but_broken = True
            break

        if not latest_turn.get("correctness", False):
            # Distinguish: did the code fail to compile, or compile-but-wrong?
            if not extraction_failed and not latest_turn.get("compiled", True):
                ended_via_compile_failure = True
            break

        if turn >= max_turns:
            ended_via_max_turns_cap = True
            break

    elapsed = time.time() - start_time

    # Compute terminal reward (logging only — no training)
    reward = terminal_correctness_only([final_trajectory_result])
    reward_value = reward[0] if reward else 0.0

    # Build output record
    turn_history_out = []
    for i, t in enumerate(accumulated_history):
        turn_history_out.append({
            "turn": i,
            "compiled": t.get("compiled"),
            "correctness": t.get("correctness"),
            "kernel_runtime_ms": t.get("kernel_runtime"),
            "reference_runtime_ms": t.get("reference_runtime"),
            "speedup": t.get("speedup"),
            "error_message": t.get("error_message"),
        })

    return {
        "problem_id": problem_id,
        "problem_name": problem_name,
        "turns_taken": termination_turn,
        "trajectory_length": len(accumulated_history),
        "final_kernel_correct": final_trajectory_result.get("final_kernel_correct", False),
        "final_kernel_runtime_ms": final_trajectory_result.get("final_kernel_runtime"),
        "turn_0_baseline_ms": turn_0_baseline.get("kernel_runtime"),
        "best_speedup_across_trajectory": final_trajectory_result.get("best_speedup_across_trajectory", 1.0),
        "termination_reason": final_trajectory_result.get("termination_reason", "unknown"),
        # Outcome flags
        "ended_via_stop_but_broken":   ended_via_stop_but_broken,
        "ended_via_max_turns_cap":     ended_via_max_turns_cap,
        "ended_via_extraction_failure": ended_via_extraction_failure,
        "ended_via_compile_failure":   ended_via_compile_failure,
        "terminal_correctness_only_reward": reward_value,
        "turn_history": turn_history_out,
        "metadata": final_trajectory_result.get("metadata", {}),
        "eval_wall_seconds": round(elapsed, 1),
    }


def _failed_problem_result(problem_id: str, problem_name: str, error: str) -> Dict[str, Any]:
    return {
        "problem_id": problem_id,
        "problem_name": problem_name,
        "turns_taken": 0,
        "trajectory_length": 0,
        "final_kernel_correct": False,
        "final_kernel_runtime_ms": None,
        "turn_0_baseline_ms": None,
        "best_speedup_across_trajectory": None,
        "termination_reason": "error",
        "ended_via_stop_but_broken":    False,
        "ended_via_max_turns_cap":      False,
        "ended_via_extraction_failure": False,
        "ended_via_compile_failure":    False,
        "terminal_correctness_only_reward": 0.0,
        "turn_history": [],
        "metadata": {"error": error},
        "eval_wall_seconds": 0.0,
    }


# ============================================================
# RESULTS WRITING
# ============================================================

def write_raw_json(problems: List[Tuple], all_results: List[Dict], path: Path,
                   smoke: bool, dry_run: bool) -> None:
    output = {
        "experiment": "drkernel14b_new_pipeline",
        "description": "hkust-nlp/drkernel-14b on cuda_iterative_optimize — pre-training baseline",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "backend": "cuda",
        "workflow": "cuda_iterative_optimize",
        "smoke_run": smoke,
        "dry_run": dry_run,
        "config": {
            "max_turns": MAX_TURNS,
            "num_correct_trials": NUM_CORRECT_TRIALS,
            "num_perf_trials": NUM_PERF_TRIALS,
            "timeout_s": TIMEOUT,
            "device": DEVICE,
            "generation": "greedy (temperature=0)",
            "max_new_tokens": 2048,
        },
        "problem_set": {
            "source": "Embedded KernelBench-style naive CUDA kernels (levels 1-2)",
            "problems": [{"id": p[0], "name": p[1]} for p in problems],
        },
        "results": all_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Raw results written: {path}")


def write_summary_md(all_results: List[Dict], path: Path,
                     smoke: bool, dry_run: bool) -> None:
    n = len(all_results)
    if n == 0:
        path.write_text("# No results.\n")
        return

    n_correct = sum(1 for r in all_results if r["final_kernel_correct"])
    correctness_rate = n_correct / n

    # Mean best speedup seen at any turn across the trajectory (correct problems only)
    best_speedups = [
        r["best_speedup_across_trajectory"]
        for r in all_results
        if r["final_kernel_correct"] and r["best_speedup_across_trajectory"] is not None
    ]
    mean_best_speedup = sum(best_speedups) / len(best_speedups) if best_speedups else None

    # Mean speedup of the final submitted kernel vs turn-0 baseline (correct problems only)
    final_speedups = [
        r["turn_0_baseline_ms"] / r["final_kernel_runtime_ms"]
        for r in all_results
        if r["final_kernel_correct"]
        and r["final_kernel_runtime_ms"] is not None
        and r["turn_0_baseline_ms"] is not None
        and r["turn_0_baseline_ms"] > 0
    ]
    mean_final_speedup = sum(final_speedups) / len(final_speedups) if final_speedups else None

    n_stop_broken  = sum(1 for r in all_results if r["ended_via_stop_but_broken"])
    n_max_turns    = sum(1 for r in all_results if r["ended_via_max_turns_cap"])
    n_extract_fail = sum(1 for r in all_results if r["ended_via_extraction_failure"])
    n_compile_fail = sum(1 for r in all_results if r.get("ended_via_compile_failure", False))

    turn_counts = [r["turns_taken"] for r in all_results if r["turns_taken"] > 0]
    turn_dist: Dict[int, int] = {}
    for t in turn_counts:
        turn_dist[t] = turn_dist.get(t, 0) + 1

    rewards = [r["terminal_correctness_only_reward"] for r in all_results]
    mean_reward = sum(rewards) / len(rewards)

    flags_note = ""
    if n_max_turns == n:
        flags_note = (
            f"\n**Dominant pattern**: {n_max_turns}/{n} trajectories ended via `max_turns_cap` "
            f"because the model never emitted `# OPTIMIZATION_STOP`. This is expected for a "
            f"pre-training baseline that has not been trained on this iterative protocol."
        )
    elif n_stop_broken > n // 2:
        flags_note = (
            f"\n**Dominant pattern**: {n_stop_broken}/{n} trajectories ended with a STOP marker "
            f"but the final kernel was broken (incorrect or failed to compile)."
        )

    lines = [
        f"# DrKernel-14B — cuda_iterative_optimize Benchmark",
        f"",
        f"**Model**: `{MODEL_ID}`  ",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Smoke run**: {smoke}  **Dry run**: {dry_run}",
        f"",
        f"## Aggregate Stats",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Problems evaluated | {n} |",
        f"| Final kernel correct | {n_correct}/{n} ({correctness_rate:.1%}) |",
        f"| Mean final-kernel speedup (correct only) | {f'{mean_final_speedup:.3f}x' if mean_final_speedup else 'N/A'} |",
        f"| Mean best speedup across trajectory (correct only) | {f'{mean_best_speedup:.3f}x' if mean_best_speedup else 'N/A'} |",
        f"| Mean terminal reward | {mean_reward:.3f} |",
        f"",
        f"## Outcome Flags",
        f"",
        f"| Flag | Count | % |",
        f"|------|-------|---|",
        f"| `ended_via_stop_but_broken` | {n_stop_broken} | {n_stop_broken/n:.0%} |",
        f"| `ended_via_max_turns_cap` | {n_max_turns} | {n_max_turns/n:.0%} |",
        f"| `ended_via_extraction_failure` | {n_extract_fail} | {n_extract_fail/n:.0%} |",
        f"| `ended_via_compile_failure` | {n_compile_fail} | {n_compile_fail/n:.0%} |",
        flags_note,
        f"",
        f"## Turn Count Distribution",
        f"",
        f"| Turns taken | Count |",
        f"|-------------|-------|",
    ]
    for t in sorted(turn_dist):
        lines.append(f"| {t} | {turn_dist[t]} |")

    lines += [
        f"",
        f"## Per-Problem Results",
        f"",
        f"| Problem | Turns | Correct | Best Speedup | Term. Reason | Reward |",
        f"|---------|-------|---------|--------------|--------------|--------|",
    ]
    for r in all_results:
        sp = f"{r['best_speedup_across_trajectory']:.3f}x" if r["best_speedup_across_trajectory"] else "N/A"
        lines.append(
            f"| {r['problem_name']} | {r['turns_taken']} "
            f"| {'✓' if r['final_kernel_correct'] else '✗'} "
            f"| {sp} "
            f"| {r['termination_reason']} "
            f"| {r['terminal_correctness_only_reward']:+.1f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Summary written: {path}")


# ============================================================
# MAIN
# ============================================================

async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate drkernel-14b on cuda_iterative_optimize")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: run first 3 problems only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stub model with fixed outputs; verify format without GPU")
    args = parser.parse_args()

    problems = SMOKE_PROBLEMS if args.smoke else ALL_PROBLEMS
    suffix = "_smoke" if args.smoke else ""

    raw_path = REPO_ROOT / "results" / f"drkernel14b_new_pipeline_raw{suffix}.json"
    summary_path = REPO_ROOT / "results" / f"drkernel14b_new_pipeline_summary{suffix}.md"

    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Problems: {len(problems)} ({'smoke' if args.smoke else 'full'})")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Max turns: {MAX_TURNS}, perf trials: {NUM_PERF_TRIALS}")

    tokenizer, model = load_model(MODEL_ID, dry_run=args.dry_run)

    scheduler = LocalDirectScheduler(dry_run=args.dry_run)
    workflow = CudaIterativeOptimizeWorkflowController()
    all_results = []

    for problem_id, problem_name, kernel_code in problems:
        try:
            result = await evaluate_problem(
                problem_id=problem_id,
                problem_name=problem_name,
                initial_kernel=kernel_code,
                workflow=workflow,
                scheduler=scheduler,
                tokenizer=tokenizer,
                model=model,
                max_turns=MAX_TURNS,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            logger.error(f"Problem {problem_id} crashed: {exc}", exc_info=True)
            result = _failed_problem_result(problem_id, problem_name, str(exc))

        all_results.append(result)
        logger.info(
            f"  >> correct={result['final_kernel_correct']}  "
            f"speedup={result['best_speedup_across_trajectory']}  "
            f"reward={result['terminal_correctness_only_reward']:+.1f}"
        )

    write_raw_json(problems, all_results, raw_path, args.smoke, args.dry_run)
    write_summary_md(all_results, summary_path, args.smoke, args.dry_run)

    # Quick summary to stdout
    n = len(all_results)
    n_correct = sum(1 for r in all_results if r["final_kernel_correct"])
    n_max_turns = sum(1 for r in all_results if r["ended_via_max_turns_cap"])
    print()
    print("=" * 60)
    print(f"DONE — {n} problems")
    print(f"  Correctness rate : {n_correct}/{n} ({n_correct/n:.0%})")
    print(f"  Max-turns cap    : {n_max_turns}/{n}")
    print(f"  Raw JSON  -> {raw_path}")
    print(f"  Summary   -> {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
