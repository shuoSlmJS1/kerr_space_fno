#!/usr/bin/env bash
set -euo pipefail

cd /home/shanjinshuo/fno_kerr/kerr_project

source /home/shanjinshuo/miniconda3/etc/profile.d/conda.sh
conda activate fno_srv

export PYTHONUNBUFFERED=1

Q_MIN=1.6007
Q_MAX=2.9993
NUM_Q=400
STEP_SIZE=0.005
DATA_SEED=20260728

generate_task() {
    local num_steps="$1"

    echo "======================================================================"
    echo "Length-Extrapolation Dataset Generation"
    echo "======================================================================"
    echo "Q range              : [${Q_MIN}, ${Q_MAX}]"
    echo "Parameter points     : ${NUM_Q}"
    echo "Lambda points        : ${num_steps}"
    echo "Step size            : ${STEP_SIZE}"
    echo "Data seed            : ${DATA_SEED}"
    echo "======================================================================"

    python -u scripts/generate_dataset.py \
      --vary-params Q \
      --Q-range "${Q_MIN}" "${Q_MAX}" \
      --sample-shape "${NUM_Q}" \
      --n-steps "${num_steps}" \
      --step-size "${STEP_SIZE}" \
      --seed "${DATA_SEED}" \
      --yes
}

echo "======================================================================"
echo "Length-Extrapolation Dataset Queue Started"
echo "======================================================================"

generate_task 1800
generate_task 2400

echo "======================================================================"
echo "Length-Extrapolation Dataset Queue Completed Successfully"
echo "======================================================================"
