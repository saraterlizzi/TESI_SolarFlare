# File: create_cache.py
import os
import yaml
from utils.dataset_h5 import DatasetH5

print("--- Inizio Script di Creazione Cache ---")

# Assumiamo che il file .yaml sia in data/solar.yaml
# Modifica il percorso se si trova altrove
config_path = 'solar.yaml' 

try:
    with open(config_path) as f:
        data_config = yaml.load(f, Loader=yaml.SafeLoader)
except FileNotFoundError:
    print(f"ERRORE: File di configurazione non trovato in {config_path}")
    exit()

train_path = data_config.get('train')
val_path = data_config.get('val')

if not train_path or not val_path:
    print("ERRORE: i percorsi 'train' o 'val' non sono definiti in solar.yaml")
    exit()

# Istanzia la classe DatasetH5 per il set di training.
# Questo avvierà il processo di caching lento, se la cache non esiste.
print(f"\nAvvio creazione cache per il set di TRAINING in: {train_path}")
DatasetH5(path=train_path)

# Istanzia la classe DatasetH5 per il set di validazione.
print(f"\nAvvio creazione cache per il set di VALIDATION in: {val_path}")
DatasetH5(path=val_path)

print("\n--- Creazione Cache Completata con Successo! ---")