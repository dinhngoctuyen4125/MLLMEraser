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

export CUDA_VISIBLE_DEVICES=0
set -euo pipefail

mkdir -p logs results

LAYER=${LAYER:-12}
STRENGTH=${STRENGTH:-1.0}
GATE_INPUT=${GATE_INPUT:-neg}

python algo.py \
  --model_id deepseek-ai/deepseek-coder-1.3b-base \
  --data_dir data/deepseek \
  --out_dir "results/L${LAYER}_t${STRENGTH}_${GATE_INPUT}" \
  --layer "$LAYER" \
  --strength "$STRENGTH" \
  --gate_train_input "$GATE_INPUT" \
  --max_test 200
