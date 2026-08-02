#!/usr/bin/env bash
set -euo pipefail

cd /home/shanjinshuo/fno_kerr/kerr_project

source /home/shanjinshuo/miniconda3/etc/profile.d/conda.sh
conda activate fno_srv

GPU_ID="${GPU_ID:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

MODEL_NAME="fno2d"
MODES_PARAM=16
MODES_LAMBDA=32
WIDTH=64
DEPTH=4
HIDDEN_DIM=128
EPOCHS=500
TRAINING_SEED=27

run_experiment() {
    local task_name="$1"
    local output_model_name="fno2d_m${MODES_PARAM}x${MODES_LAMBDA}_w${WIDTH}_d${DEPTH}_e${EPOCHS}"

    echo "======================================================================"
    echo "Extended-Epoch Training"
    echo "======================================================================"
    echo "Physical GPU         : ${GPU_ID}"
    echo "Task                 : ${task_name}"
    echo "Model                : ${output_model_name}"
    echo "Epochs               : ${EPOCHS}"
    echo "======================================================================"

    python -u scripts/train_model_2d.py \
      --task-name "${task_name}" \
      --model "${MODEL_NAME}" \
      --modes-param "${MODES_PARAM}" \
      --modes-lambda "${MODES_LAMBDA}" \
      --width "${WIDTH}" \
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
    echo "Original-Test Evaluation"
    echo "======================================================================"
    echo "Task                 : ${task_name}"
    echo "Model                : ${output_model_name}"
    echo "======================================================================"

    python -u scripts/run_analysis_2d.py \
      --task-name "${task_name}" \
      --model-name "${output_model_name}" \
      --device cuda \
      --batch-size 1

    python - "${task_name}" "${output_model_name}" <<'PY'
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

if not summary_path.exists():
    raise FileNotFoundError(
        f"Missing summary file: {summary_path}"
    )

with summary_path.open("r", encoding="utf-8") as file:
    summary = json.load(file)

physical = summary["metrics"]["physical_space"]

if not isinstance(physical, dict):
    raise RuntimeError(
        f"Physical-space metrics are missing: {summary_path}"
    )

print("Verified completed run")
print(f"Task                  : {task_name}")
print(f"Model                 : {model_name}")
print(
    "Physical-space MSE    : "
    f"{physical['test_mse']:.6e}"
)
print(
    "Physical-space RelL2  : "
    f"{physical['test_relative_l2']:.6e}"
)
PY
}

echo "======================================================================"
echo "Extended-Epoch Experiment Queue Started"
echo "======================================================================"

run_experiment "q_1p6-3_n2000_t1200"
run_experiment "q_1p6-3_n5000_t1200"

echo "======================================================================"
echo "All Extended-Epoch Experiments Completed Successfully"
echo "======================================================================"
