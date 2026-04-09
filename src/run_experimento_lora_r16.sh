#!/bin/bash
#SBATCH --job-name=experimento_sedici_lora_r16
#SBATCH --output=logs_%x_%j.out
#SBATCH --error=logs_%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

module purge

eval "$(micromamba shell hook --shell bash)"
micromamba activate experimento

# Variables de entorno para ejecución estricta offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

MODEL_PATH="/path/model"

echo "========================================"
echo "          FASE 1: ENTRENAMIENTO         "
echo "========================================"

echo "Entrenando LoRA r=16..."
python experimento_sedici.py --model_path $MODEL_PATH --method lora --lora_r 16 --fase train

echo "========================================"
echo "          FASE 2: EVALUACIóN	      "
echo "========================================"

echo "Evaluando LoRA r=16..."
python experimento_sedici.py --model_path $MODEL_PATH --method lora --lora_r 16 --fase eval
