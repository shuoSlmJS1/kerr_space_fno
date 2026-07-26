#!/usr/bin/env bash
set -euo pipefail

cd /home/shanjinshuo/fno_kerr/kerr_project

source /home/shanjinshuo/miniconda3/etc/profile.d/conda.sh
conda activate fno_srv

DATA_SEED=10
TRAINING_SEED=27

Q_MIN=1.6
Q_MAX=3.0

N_STEPS=1200
STEP_SIZE=0.005

MODES_PARAM=16
MODES_LAMBDA=32
DEPTH=4
HIDDEN_DIM=128
EPOCHS=300

run_dataset() {
    local n="$1"
    local task_name="q_1p6-3_n${n}_t${N_STEPS}"

    echo "======================================================================"
    echo "Dataset Generation"
    echo "Task                 : ${task_name}"
    echo "Target samples       : ${n}"
    echo "Lambda points        : ${N_STEPS}"
    echo "======================================================================"

    python scripts/generate_dataset.py \
      --vary-params Q \
      --Q-range "${Q_MIN}" "${Q_MAX}" \
      --sample-shape "${n}" \
      --n-steps "${N_STEPS}" \
      --step-size "${STEP_SIZE}" \
      --seed "${DATA_SEED}" \
      --yes
}

run_model() {
    local n="$1"
    local width="$2"

    local task_name="q_1p6-3_n${n}_t${N_STEPS}"
    local model_name="fno2d_m${MODES_PARAM}x${MODES_LAMBDA}_w${width}_d${DEPTH}_e${EPOCHS}"

    echo "======================================================================"
    echo "Model Training"
    echo "Task                 : ${task_name}"
    echo "Model                : ${model_name}"
    echo "======================================================================"

    python scripts/train_model_2d.py \
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

    echo "======================================================================"
    echo "Model Evaluation"
    echo "Task                 : ${task_name}"
    echo "Model                : ${model_name}"
    echo "======================================================================"

    python scripts/run_analysis_2d.py \
      --task-name "${task_name}" \
      --model-name "${model_name}" \
      --device cuda \
      --batch-size 1
}

echo "======================================================================"
echo "2D Scale Experiment Queue Started"
echo "======================================================================"

# 数据集
run_dataset 500
run_dataset 1000
run_dataset 2000
run_dataset 5000

# 数据量规模实验：固定 width=64
run_model 500 64
run_model 1000 64
run_model 2000 64
run_model 5000 64

# 模型规模实验：固定 n=5000
# width=64 已经在上一组完成，因此不重复运行。
run_model 5000 32
run_model 5000 96

echo "======================================================================"
echo "All 2D Scale Experiments Completed Successfully"
echo "======================================================================"
