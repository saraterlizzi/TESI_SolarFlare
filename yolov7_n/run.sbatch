#!/bin/bash
#SBATCH --job-name=SF_Train  
#SBATCH --output=gpu.out 
#SBATCH --error=gpu.err  
#SBATCH --nodes=1        
#SBATCH --partition=xgpu
#SBATCH --ntasks=8       
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=44000 

cd /home/s.terlizzi/SolarFlare/yolov7_n/

module load cuda/11.6

source /home/s.terlizzi/miniconda3/etc/profile.d/conda.sh

conda activate SolarFlare

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python train.py    --resume    --workers 4   --device 0   --batch-size 32  --data solar.yaml   --img 640 640   --cfg cfg/training/yolov7.yaml   --weights ''   --name yolov7_h_custom   --epochs 300 --hyp hyp.yaml
