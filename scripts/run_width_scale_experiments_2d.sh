#!/usr/bin/env bash
set -euo pipefail

cd /home/shanjinshuo/fno_kerr/kerr_project

source /home/shanjinshuo/miniconda3/etc/profile.d/conda.sh
conda activate fno_srv

GPU_ID="${GPU_ID:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_NAME="q_1p6-3_n5000_t1200"

MODES_PARAM=16
MODES_LAMBDA=32
DEPTH=4
HIDDEN_DIM=128
EPOCHS=300
TRAINING_SEED=27

run_model() {
    local width="$1"
    local model_name="fno2d_m${MODES_PARAM}x${MODES_LAMBDA}_w${width}_d${DEPTH}_e${EPOCHS}"

    echo "======================================================================"
    echo "Width-Scale Model Training"
    echo "Physical GPU         : ${GPU_ID}"
    echo "Task                 : ${TASK_NAME}"
    echo "Model                : ${model_name}"
    echo "======================================================================"

    python scripts/train_model_2d.py \
      --task-name "${TASK_NAME}" \
      --model fno2d \
      --modes-param "${MODES_PARAM}" \
      --modes-lambda "${MODES_LAMBDA}" \
      --width "${width}" \
      --depth "${DEPTH}" \
      --hidden-dim "${HIDDEN_DIM}" \
      --normalization standard \
      --target-transform raw \
      --batch-size 1 \
      --epochs "${EPOCHS}" \
      --lr 0.001 \
      --weight-decay 0.0001 \
      --scheduler-gamma 0.995 \
      --training-seed "${TRAINING_SEED}" \
      --device cuda \
      --print-every 10

    echo "======================================================================"
    echo "Width-Scale Model Evaluation"
    echo "Task                 : ${TASK_NAME}"
    echo "Model                : ${model_name}"
    echo "======================================================================"

    python scripts/run_analysis_2d.py \
      --task-name "${TASK_NAME}" \
      --model-name "${model_name}" \
      --device cuda \
      --batch-size 1

    python - "${TASK_NAME}" "${model_name}" <<'PY'
import json
import sys
from pathlib import Path

task_name = sys.argv[1]
model_name = sys.argv[2]

summary_path = (
    Path("outputs")
    / task_name
    / model_name
    / "summary.json"
)

with summary_path.open("r", encoding="utf-8") as f:
    summary = json.load(f)

physical = summary["metrics"]["physical_space"]

if physical is None:
    raise RuntimeError(
        f"Physical-space evaluation is missing: {summary_path}"
    )

if not summary.get("evaluation_completed", False):
    raise RuntimeError(
        f"Evaluation is not marked complete: {summary_path}"
    )

print(
    "Verified official Relative L2: "
    f"{physical['test_relative_l2']:.6e}"
)
PY
}

echo "======================================================================"
echo "2D Width-Scale Experiment Queue Started"
echo "======================================================================"

run_model 16
run_model 48
run_model 80

echo "======================================================================"
echo "All 2D Width-Scale Experiments Completed Successfully"
echo "======================================================================"
