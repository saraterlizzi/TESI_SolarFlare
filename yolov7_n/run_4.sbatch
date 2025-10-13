#!/bin/bash
#SBATCH --job-name=gpu_4GPU
#SBATCH --output=gpu_4GPU.out  
#SBATCH --error=gpu_4GPU.err
#SBATCH --nodes=1              
#SBATCH --partition=xgpu
#SBATCH --gres=gpu:tesla:4      
#SBATCH --time=06:00:00

# Spostati nella cartella corretta
cd /home/s.terlizzi/SolarFlare/yolov7_n/

# Carica i moduli di sistema
module load cuda/11.6

# Inizializza e attiva l'ambiente Conda
source /home/s.terlizzi/miniconda3/etc/profile.d/conda.sh
conda activate SolarFlare

# Imposta le variabili d'ambiente
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Esegui lo script in modalità distribuita
torchrun --nproc_per_node=4 --master_port 9527 train.py \
  --workers 4 \
  --sync-bn \
  --batch-size 64 \
  --data solar.yaml \
  --img 640 640 \
  --cfg cfg/training/yolov7.yaml \
  --weights '' \
  --name yolov7_h_custom_4GPU \
  --epochs 300 \
  --hyp hyp.yaml
