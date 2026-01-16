import cv2
import numpy as np
import os
import glob

# --- Impostazioni ---
CARTELLA_INPUT = 'Validation/images_original'
CARTELLA_OUTPUT = 'Validation/images'

# Percentuale del raggio. 
FATTORE_RAGGIO = 0.96 

# Creazione della cartella di output se non esiste
if not os.path.exists(CARTELLA_OUTPUT):
    os.makedirs(CARTELLA_OUTPUT)
    print(f"Cartella '{CARTELLA_OUTPUT}' creata.")

# Ricerca di tutti i file immagine (anche nelle sottocartelle)
tipi_file = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif')
file_immagini = []
for tipo in tipi_file:
    file_immagini.extend(glob.glob(os.path.join(CARTELLA_INPUT, '**', tipo), recursive=True))

if not file_immagini:
    print(f"Nessuna immagine trovata nella cartella '{CARTELLA_INPUT}'.")
    print("Assicurati di aver creato una cartella 'input' e di averci messo le tue immagini.")
else:
    print(f"Trovate {len(file_immagini)} immagini. Inizio elaborazione...")

# Elaborazione di ogni immagine
for percorso_immagine in file_immagini:
    # Caricamento dell'immagine originale
    img_originale = cv2.imread(percorso_immagine)
    
    if img_originale is None:
        print(f"Errore: Impossibile caricare l'immagine {percorso_immagine}")
        continue
        
    # Dimensioni dell'immagine
    h, w = img_originale.shape[:2]

    # Calcolo del centro e del raggio
    centro_x = w // 2
    centro_y = h // 2
    
    # Calcolo del raggio basato sulla dimensione più piccola
    # e applicazione del fattore per escludere i bordi
    raggio_base = min(centro_x, centro_y)
    raggio = int(raggio_base * FATTORE_RAGGIO)

    # 1. Creazione di una maschera completamente nera
    # (della stessa dimensione e tipo dell'originale)
    maschera = np.zeros_like(img_originale)

    # 2. Disegno di un cerchio pieno bianco sulla maschera (area da conservare)
    cv2.circle(maschera, (centro_x, centro_y), raggio, (255, 255, 255), thickness=cv2.FILLED)

    # 3. Applicazione della maschera all'immagine originale
    risultato = cv2.bitwise_and(img_originale, maschera)

    # Costruzione del percorso di output mantenendo la struttura
    percorso_relativo = os.path.relpath(percorso_immagine, CARTELLA_INPUT)
    percorso_output = os.path.join(CARTELLA_OUTPUT, percorso_relativo)
    
    cartella_destinazione = os.path.dirname(percorso_output)
    if not os.path.exists(cartella_destinazione):
        os.makedirs(cartella_destinazione)

    # Salvataggio dell'immagine modificata
    cv2.imwrite(percorso_output, risultato)
    
print(f"\nElaborazione completata. Immagini salvate in '{CARTELLA_OUTPUT}'.")