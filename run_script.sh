#!/bin/bash

#SBATCH --job-name=mllme
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail
# Respect the GPU selection supplied by the shell or scheduler.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

mkdir -p logs results

LAYER=${LAYER:-12}
STRENGTH=${STRENGTH:-1.0}
GATE_INPUT=${GATE_INPUT:-neg}
EXTRACT_BS=${EXTRACT_BS:-32}
GEN_BS=${GEN_BS:-32}

python algo.py \
  --model_id deepseek-ai/deepseek-coder-1.3b-instruct \
  --data_dir ../Data-Collection/deepseek \
  --out_dir "results/L${LAYER}_t${STRENGTH}_${GATE_INPUT}" \
  --layer "$LAYER" \
  --strength "$STRENGTH" \
  --gate_train_input "$GATE_INPUT" \
  --extract_bs "$EXTRACT_BS" \
  --gen_bs "$GEN_BS" \
  --max_test 200
