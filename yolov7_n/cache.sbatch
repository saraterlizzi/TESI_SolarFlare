#!/bin/bash
#SBATCH --job-name=create_cache
#SBATCH --output=cache.out
#SBATCH --error=cache.err
#SBATCH --nodes=1               # Richiede 1 solo computer, come specificato.
#SBATCH --partition=xhicpu      # OTTIMIZZAZIONE: Usa una partizione CPU, ideale per il caching.
#SBATCH --ntasks=1              # Esegue 1 solo processo per evitare conflitti sul disco.
#SBATCH --cpus-per-task=8       # Assegna 8 core CPU a quel processo per accelerarlo.
#SBATCH --mem=64G               # Assegna 64GB di RAM per sicurezza.
#SBATCH --time=12:00:00         # Imposta il tempo massimo a 12 ore, come specificato.

# --- INIZIO SCRIPT ---

echo "Job iniziato in data: $(date)"

# 1. Spostati nella cartella corretta del progetto
cd /home/s.terlizzi/SolarFlare/yolov7_n/

# 2. Inizializza e attiva l'ambiente Conda
source /home/s.terlizzi/miniconda3/etc/profile.d/conda.sh
conda activate SolarFlare

# 3. Esegui lo script di creazione cache
python create_cache.py

echo "Job terminato in data: $(date)"