# File: track_metrics_fix.py

import argparse
import os
import glob
import shutil
import numpy as np
import torch
import cv2
import h5py
from tqdm import tqdm

# Import Norfair
import norfair
from norfair import Detection, Tracker
from norfair.metrics import Accumulators, InformationFile 

from models.experimental import attempt_load
from utils.general import non_max_suppression

# --- 1. UTILITY: Prepara il GT personalizzato per Norfair ---
def setup_custom_mot_structure(custom_gt_path, img_list, output_base, img_w, img_h):
    """
    Legge il TUO file ground_truth e crea la struttura di cartelle 
    standard MOTChallenge richiesta dalle metriche.
    """
    gt_dir = os.path.join(output_base, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    
    dest_gt_path = os.path.join(gt_dir, "gt.txt")
    
    print(f"Preparazione Ground Truth da: {custom_gt_path}")
    
    if not os.path.exists(custom_gt_path):
        print(f"ERRORE CRITICO: Il file GT {custom_gt_path} non esiste!")
        exit()

    with open(custom_gt_path, "r") as in_f, open(dest_gt_path, "w") as out_f:
        for line in in_f:
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            
            try:
                frame = int(parts[0])
                obj_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                
                # Formato MOT: frame, id, left, top, width, height, conf, -1, -1, -1
                mot_line = f"{frame},{obj_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
                out_f.write(mot_line)
            except ValueError:
                continue

    # Creazione seqinfo.ini
    seq_len = len(img_list)
    seqinfo_path = os.path.join(output_base, "seqinfo.ini")
    with open(seqinfo_path, "w") as f:
        f.write("[Sequence]\n")
        f.write(f"name={os.path.basename(output_base)}\n")
        f.write(f"imDir=img1\n")
        f.write(f"frameRate=30\n")
        f.write(f"seqLength={seq_len}\n")
        f.write(f"imWidth={img_w}\n")
        f.write(f"imHeight={img_h}\n")
        f.write(f"imExt=.jpg\n")

# --- 2. CONVERSIONE YOLO -> NORFAIR ---
def yolo_to_norfair(yolo_preds):
    norfair_detections = []
    for det in yolo_preds:
        bbox = np.array([
            [det[0].item(), det[1].item()],
            [det[2].item(), det[3].item()]
        ])
        scores = np.array([det[4].item(), det[4].item()])
        label = int(det[5].item())
        norfair_detections.append(Detection(points=bbox, scores=scores, label=label))
    return norfair_detections

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Cartella immagini")
    parser.add_argument("--gt-file", type=str, required=True, help="File ground_truth_auto.txt")
    parser.add_argument("--model-path", type=str, required=True, help="Modello .pt")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Caricamento Modello (Gestione JIT/Torchscript)
    print(f"Caricamento modello: {args.model_path}")
    try:
        model = attempt_load(args.model_path, map_location=device)
    except (AttributeError, RuntimeError):
        print("Rilevato modello JIT/Traced. Uso torch.jit.load...")
        model = torch.jit.load(args.model_path, map_location=device)
    model.eval()

    # 2. Lista File
    input_files = sorted(glob.glob(os.path.join(args.source, '*.*')))
    input_files = [f for f in input_files if f.endswith(('.jpg', '.jpeg', '.png', '.h5'))]
    if not input_files:
        print("ERRORE: Nessun file trovato.")
        exit()

    # Cartella temporanea
    TEMP_MOT_DIR = "temp_metrics_env"
    if os.path.exists(TEMP_MOT_DIR): shutil.rmtree(TEMP_MOT_DIR)
    os.makedirs(TEMP_MOT_DIR, exist_ok=True)

    # 3. Setup Ambiente Metriche
    setup_custom_mot_structure(args.gt_file, input_files, TEMP_MOT_DIR, args.img_size, args.img_size)

    # 4. Inizializza Tracker
    tracker = Tracker(
        distance_function="iou", 
        distance_threshold=0.7,
        detection_threshold=args.conf_thres
    )
    
    # --- FIX CRITICO ---
    # Creazione oggetto InformationFile prima di passarlo
    print("Inizializzazione Metriche...")
    seqinfo_path = os.path.join(TEMP_MOT_DIR, "seqinfo.ini")
    
    try:
        # Creazione oggetto che Norfair vuole
        info_file = InformationFile(file_path=seqinfo_path)
        
        acc = Accumulators()
        # Passaggio oggetto, NON la stringa
        acc.create_accumulator(input_path=TEMP_MOT_DIR, information_file=info_file)
        
    except Exception as e:
        print(f"Errore inizializzazione Norfair: {e}")
        exit()

    print(f"\nAvvio Tracking su {len(input_files)} frames...")
    
    # LOOP
    for i, path in enumerate(tqdm(input_files)):
        # Load Img
        if path.endswith('.h5'):
            with h5py.File(path, 'r') as f: data = np.nan_to_num(f['magnetogram/data'][:])
            norm = (np.clip(data, -1500, 1500) + 1500) / 3000
            img0 = cv2.resize(norm, (args.img_size, args.img_size))
            img_tensor = torch.from_numpy(np.stack([img0]*3, axis=-1)).float()
        else:
            img0 = cv2.imread(path)
            if img0 is None: continue
            img0 = cv2.resize(img0, (args.img_size, args.img_size))
            img_tensor = torch.from_numpy(img0 / 255.0).float()

        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # Inferenza
        with torch.no_grad():
            try:
                pred = model(img_tensor, augment=False)[0]
            except:
                pred = model(img_tensor)[0] # Fallback JIT
        
        pred = non_max_suppression(pred, args.conf_thres)[0]

        # Tracker Update
        if pred is not None and len(pred) > 0:
            dets = yolo_to_norfair(pred)
            tracked_objs = tracker.update(detections=dets)
        else:
            tracked_objs = tracker.update(detections=[])

        # Metriche Update
        acc.update(predictions=tracked_objs)

    # 5. Risultati
    print("\n" + "="*40)
    print(" RISULTATI FINALI ")
    print("="*40)
    
    try:
        metrics = acc.compute_metrics()
        print(metrics)
        
        os.makedirs("risultati_finali", exist_ok=True)
        acc.save_metrics(save_path="risultati_finali", file_name="report_metriche_complete.txt")
        print("\nReport salvato in: risultati_finali/report_metriche_last.txt")
        
    except Exception as e:
        print(f"\nERRORE calcolo metriche: {e}")