#!/bin/bash
#SBATCH --job-name=eval_lora_r16_sedici_fft
#SBATCH --output=logs_%x_%j.out
#SBATCH --error=logs_%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module purge

eval "$(micromamba shell hook --shell bash)"
micromamba activate experimento

# Variables de entorno para ejecución estricta offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

MODEL_PATH="/path/model"

echo "========================================"
echo "          FASE 2: EVALUACIOóN	      "
echo "========================================"

echo "Evaluando FFT..."
python evaluacion_sedici.py --model_path $MODEL_PATH --method lora --lora_r 16
