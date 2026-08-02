#!/usr/bin/env bash
set -euo pipefail

cd /home/shanjinshuo/fno_kerr/kerr_project

source /home/shanjinshuo/miniconda3/etc/profile.d/conda.sh
conda activate fno_srv

GPU_ID="${GPU_ID:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

MODES_PARAM=16
MODES_LAMBDA=32
DEPTH=4
EPOCHS=500
HIDDEN_DIM=128
TRAINING_SEED=27

COMMON_TASK_NAME="q_1p6007-2p9993_n400_t1200"

run_is_complete() {
    local task_name="$1"
    local model_name="$2"

    local summary_path="outputs/${task_name}/${model_name}/summary.json"

    if [[ ! -f "${summary_path}" ]]; then
        return 1
    fi

    python - "${summary_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

with path.open("r", encoding="utf-8") as file:
    summary = json.load(file)

run_completed = summary.get("run_completed") is True
evaluation_completed = summary.get("evaluation_completed") is True
physical = summary.get("metrics", {}).get("physical_space")

if run_completed and evaluation_completed and isinstance(physical, dict):
    raise SystemExit(0)

raise SystemExit(1)
PY
}

run_experiment() {
    local task_name="$1"
    local width="$2"

    local model_name
    model_name="fno2d_m${MODES_PARAM}x${MODES_LAMBDA}_w${width}_d${DEPTH}_e${EPOCHS}"

    echo "======================================================================"
    echo "Queue A Experiment"
    echo "======================================================================"
    echo "Physical GPU         : ${GPU_ID}"
    echo "Task                 : ${task_name}"
    echo "Model                : ${model_name}"
    echo "Modes                : ${MODES_PARAM} x ${MODES_LAMBDA}"
    echo "Width                : ${width}"
    echo "Depth                : ${DEPTH}"
    echo "Epochs               : ${EPOCHS}"
    echo "======================================================================"

    if run_is_complete "${task_name}" "${model_name}"; then
        echo "Completed result already exists. Skipping training and original-test analysis."
    else
        python -u scripts/train_model_2d.py \
          --task-name "${task_name}" \
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

        python -u scripts/run_analysis_2d.py \
          --task-name "${task_name}" \
          --model-name "${model_name}" \
          --device cuda \
          --batch-size 1
    fi

    echo "----------------------------------------------------------------------"
    echo "Common-Test Evaluation"
    echo "----------------------------------------------------------------------"

    python -u scripts/evaluate_common_test_2d.py \
      --common-task-name "${COMMON_TASK_NAME}" \
      --model-task-names "${task_name}" \
      --model-name "${model_name}" \
      --device cuda \
      --output-name "queue_a__${task_name}__${model_name}"

    echo "----------------------------------------------------------------------"
    echo "Experiment Completed"
    echo "Task                 : ${task_name}"
    echo "Model                : ${model_name}"
    echo "----------------------------------------------------------------------"
}

echo "======================================================================"
echo "Queue A Cross-Scale Experiments Started"
echo "======================================================================"

# ----------------------------------------------------------------------
# A. width=48 数据量曲线
# ----------------------------------------------------------------------
run_experiment "q_1p6-3_n500_t1200" 48
run_experiment "q_1p6-3_n1000_t1200" 48
run_experiment "q_1p6-3_n2000_t1200" 48
run_experiment "q_1p6-3_n5000_t1200" 48

# ----------------------------------------------------------------------
# B. n=2000 宽度曲线
# n2000/w48 已在上方运行。
# n2000/w64/e500 已存在，不在此处重跑。
# ----------------------------------------------------------------------
run_experiment "q_1p6-3_n2000_t1200" 16
run_experiment "q_1p6-3_n2000_t1200" 32
run_experiment "q_1p6-3_n2000_t1200" 80

echo "======================================================================"
echo "Queue A Cross-Scale Experiments Completed Successfully"
echo "======================================================================"
