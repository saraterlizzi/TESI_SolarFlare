# File: preprocess_to_zip.py

import h5py
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
import argparse
import sys
import zipfile
import concurrent.futures

# --- Impostazioni ---
IMG_SIZE = 640
CLIP_MIN, CLIP_MAX = -1500, 1500
CLASS_ID = 0
N_WORKERS = 32 

def process_file_h5(h5_path):
    """
    Processa un singolo file H5 e RESTITUISCE i dati pronti per lo zip.
    Questa funzione è progettata per essere eseguita in un thread separato.
    """
    base_name = os.path.splitext(os.path.basename(h5_path))[0]
    img_arcname = os.path.join('images', f"{base_name}.jpg")
    label_arcname = os.path.join('labels', f"{base_name}.txt")

    try:
        with h5py.File(h5_path, 'r') as f:
            
            # --- 1. Estrazione Dati Immagine ---
            magnetogram_data = f['magnetogram/data']
            orig_h, orig_w = magnetogram_data.shape
            data = magnetogram_data[:]

            # --- 2. Estrazione Etichette (Bounding Box) ---
            harp_group = f['harp/metadata']
            image_labels = []
            
            for harp_id in harp_group:
                harp_attrs = harp_group[harp_id].attrs
                
                required_keys = ['CRPIX1', 'CRPIX2', 'CRSIZE1', 'CRSIZE2']
                if not all(key in harp_attrs for key in required_keys):
                    continue

                w_abs = float(harp_attrs['CRSIZE1'])
                h_abs = float(harp_attrs['CRSIZE2'])

                if w_abs <= 0 or h_abs <= 0:
                    continue

                x_center_norm = (float(harp_attrs['CRPIX1']) + w_abs / 2) / orig_w
                y_center_norm = (float(harp_attrs['CRPIX2']) + h_abs / 2) / orig_h
                width_norm = w_abs / orig_w
                height_norm = h_abs / orig_h

                if not (0.0 < x_center_norm < 1.0 and 0.0 < y_center_norm < 1.0):
                    continue
                
                image_labels.append([CLASS_ID, x_center_norm, y_center_norm, width_norm, height_norm])

            # --- 3. Processing Immagine ---
            if np.isnan(data).any() or np.isinf(data).any():
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            clipped_data = np.clip(data, CLIP_MIN, CLIP_MAX)
            normalized_data = (clipped_data - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)
            resized_image = cv2.resize(normalized_data, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            image_uint8 = (resized_image * 255.0).astype(np.uint8)
            image_rgb = np.stack([image_uint8] * 3, axis=-1)
            
            # Codifica in memoria
            is_success, img_buffer = cv2.imencode('.jpg', image_rgb)
            if not is_success:
                raise Exception("Impossibile codificare l'immagine in JPG.")
            
            # --- 4. Preparazione Etichette ---
            label_lines = [f"{lbl[0]} {lbl[1]} {lbl[2]} {lbl[3]} {lbl[4]}" for lbl in image_labels]
            label_content = "\n".join(label_lines)
            
            # Restituzione del necessario per la scrittura
            return (img_arcname, img_buffer.tobytes(), label_arcname, label_content)

    except Exception as e:
        print(f"\nATTENZIONE: Fallimento nel processare {h5_path}: {e}")
        # Creazione dati placeholder
        placeholder_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        is_success, img_buffer = cv2.imencode('.jpg', placeholder_img)
        return (img_arcname, img_buffer.tobytes(), label_arcname, "")


def main():
    parser = argparse.ArgumentParser(description="Converte il dataset H5 in un singolo file .zip in formato YOLO (multithread).")
    parser.add_argument('--source-dir', type=str, required=True, help="Cartella locale contenente i file .h5")
    parser.add_argument('--zip-file', type=str, required=True, help="Percorso del file .zip di output")
    parser.add_argument('--workers', type=int, default=N_WORKERS, help="Numero di thread 'produttori'")
    args = parser.parse_args()

    # Ricerca di tutti i file H5
    h5_files = sorted(glob.glob(os.path.join(args.source_dir, '**', '*.h5'), recursive=True))
    if not h5_files:
        print(f"Errore: Nessun file .h5 trovato in {args.source_dir}")
        sys.exit(1)
        
    print(f"Trovati {len(h5_files)} file .h5. Avvio della conversione in '{args.zip_file}'...")
    print(f"Uso di {args.workers} thread lavoratori.")

    # Il ThreadPoolExecutor gestisce il pool di thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        
        # Apertura del file Zip 
        with zipfile.ZipFile(args.zip_file, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            
            # Sottomissione di tutti i lavori al pool di thread
            futures = [executor.submit(process_file_h5, h5_path) for h5_path in h5_files]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(h5_files), desc="Conversione in corso"):
                
                # Risultato (img_arcname, img_buffer, label_arcname, label_content)
                result = future.result()
                
                if result:
                    img_arcname, img_buffer, label_arcname, label_content = result
                    
                    # Scrittura dei risultati nello zip
                    zf.writestr(img_arcname, img_buffer)
                    zf.writestr(label_arcname, label_content)

    print("\nConversione completata.")
    print(f"Il file '{args.zip_file}' è stato creato con successo.")

if __name__ == "__main__":
    main()