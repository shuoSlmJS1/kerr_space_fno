#!/usr/bin/env bash
set -euo pipefail

cd /home/shanjinshuo/fno_kerr/kerr_project

source /home/shanjinshuo/miniconda3/etc/profile.d/conda.sh
conda activate fno_srv

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

TASK_NAME="q_1p6-3_n2000_t1200"
COMMON_TASK_NAME="q_1p6007-2p9993_n400_t1200"

MODES_PARAM=16
MODES_LAMBDA=32
WIDTH=64
EPOCHS=500
HIDDEN_DIM=128
TRAINING_SEED=27

run_experiment() {
    local depth="$1"
    local model_name="fno2d_m${MODES_PARAM}x${MODES_LAMBDA}_w${WIDTH}_d${depth}_e${EPOCHS}"

    echo "======================================================================"
    echo "Depth Experiment"
    echo "======================================================================"
    echo "Physical GPU : ${GPU_ID}"
    echo "Task         : ${TASK_NAME}"
    echo "Model        : ${model_name}"
    echo "Modes        : ${MODES_PARAM} x ${MODES_LAMBDA}"
    echo "Width        : ${WIDTH}"
    echo "Depth        : ${depth}"
    echo "Epochs       : ${EPOCHS}"
    echo "======================================================================"

    python -u scripts/train_model_2d.py \
      --task-name "${TASK_NAME}" \
      --model fno2d \
      --modes-param "${MODES_PARAM}" \
      --modes-lambda "${MODES_LAMBDA}" \
      --width "${WIDTH}" \
      --depth "${depth}" \
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
      --task-name "${TASK_NAME}" \
      --model-name "${model_name}" \
      --device cuda \
      --batch-size 1

    python -u scripts/evaluate_common_test_2d.py \
      --common-task-name "${COMMON_TASK_NAME}" \
      --model-task-names "${TASK_NAME}" \
      --model-name "${model_name}" \
      --device cuda \
      --output-name "depth_scale__${model_name}"

    echo "Completed: ${model_name}"
}

# depth=4 已有正式结果，不重跑。
run_experiment 2
run_experiment 3
run_experiment 5
run_experiment 6

echo "======================================================================"
echo "All depth experiments completed successfully."
echo "======================================================================"
