#!/usr/bin/env bash
set -euo pipefail

cd /home/shanjinshuo/fno_kerr/kerr_project
source /home/shanjinshuo/miniconda3/etc/profile.d/conda.sh
conda activate fno_srv

run_visualization() {
    local task_name="$1"
    local model_name="$2"
    local label="$3"

    echo "======================================================================"
    echo "Generating presentation HTML"
    echo "Task  : ${task_name}"
    echo "Model : ${model_name}"
    echo "Label : ${label}"
    echo "======================================================================"

    python scripts/analyze_fno2d_error_and_3d.py \
      --task-name "${task_name}" \
      --model-name "${model_name}" \
      --label "${label}"
}

# width=48 数据量曲线
run_visualization \
  "q_1p6-3_n500_t1200" \
  "fno2d_m16x32_w48_d4_e500" \
  "Q-only FNO2D | n=500 | width=48"

run_visualization \
  "q_1p6-3_n1000_t1200" \
  "fno2d_m16x32_w48_d4_e500" \
  "Q-only FNO2D | n=1000 | width=48"

run_visualization \
  "q_1p6-3_n2000_t1200" \
  "fno2d_m16x32_w48_d4_e500" \
  "Q-only FNO2D | n=2000 | width=48"

run_visualization \
  "q_1p6-3_n5000_t1200" \
  "fno2d_m16x32_w48_d4_e500" \
  "Q-only FNO2D | n=5000 | width=48"

# n=2000 宽度曲线
run_visualization \
  "q_1p6-3_n2000_t1200" \
  "fno2d_m16x32_w16_d4_e500" \
  "Q-only FNO2D | n=2000 | width=16"

run_visualization \
  "q_1p6-3_n2000_t1200" \
  "fno2d_m16x32_w32_d4_e500" \
  "Q-only FNO2D | n=2000 | width=32"

run_visualization \
  "q_1p6-3_n2000_t1200" \
  "fno2d_m16x32_w64_d4_e500" \
  "Q-only FNO2D | n=2000 | width=64"

run_visualization \
  "q_1p6-3_n2000_t1200" \
  "fno2d_m16x32_w80_d4_e500" \
  "Q-only FNO2D | n=2000 | width=80"

echo "======================================================================"
echo "All presentation HTML files were generated successfully."
echo "======================================================================"
