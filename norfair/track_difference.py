# File: track.py

import argparse
import os
from typing import List
import subprocess

import numpy as np
import torch
import cv2
import h5py
import glob
from tqdm import tqdm

import norfair
from norfair import Detection, Tracker

# Importa le funzioni necessarie da YOLOv7
from models.experimental import attempt_load
from utils.general import non_max_suppression

# --- FUNZIONI DI UTILITÀ ---
def yolo_detections_to_norfair_detections(yolo_detections: torch.tensor) -> List[Detection]:
    """Converte le rilevazioni di YOLOv7 in un formato comprensibile da Norfair."""
    norfair_detections: List[Detection] = []
    
    for detection in yolo_detections:
        bbox = np.array(
            [
                [detection[0].item(), detection[1].item()],
                [detection[2].item(), detection[3].item()],
            ]
        )
        scores = np.array([detection[4].item(), detection[4].item()])
        norfair_detections.append(
            Detection(points=bbox, scores=scores, label=int(detection[5].item()))
        )

    return norfair_detections

def load_ground_truth(label_path, img_size):
    """Legge le etichette reali (Ground Truth) da un file .txt in formato YOLO."""
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            x_c *= img_size
            y_c *= img_size
            w *= img_size
            h *= img_size

            x_min = int(x_c - w / 2)
            y_min = int(y_c - h / 2)
            x_max = int(x_c + w / 2)
            y_max = int(y_c + h / 2)

            gt_boxes.append([x_min, y_min, x_max, y_max])
    
    return gt_boxes

def get_label_path_from_image(image_path, labels_dir_arg):
    """Determina il percorso del file label dato il percorso immagine."""
    base_filename = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(base_filename)[0]
    search_dir = labels_dir_arg if labels_dir_arg else os.path.dirname(image_path)
    return os.path.join(search_dir, filename_no_ext + ".txt")

# --- INIZIO SCRIPT PRINCIPALE ---
# 1. DEFINIZIONE DEGLI ARGOMENTI
parser = argparse.ArgumentParser(description="Traccia le regioni attive solari (Standard).")
parser.add_argument("--input-dir", type=str, required=True, help="Cartella input (.h5 o .jpg).")
parser.add_argument("--labels-dir", type=str, default=None, help="Cartella labels (opzionale).")
parser.add_argument("--model-path", type=str, required=True, help="Modello .pt.")
parser.add_argument("--img-size", type=int, default=640, help="Dimensione immagine.")
parser.add_argument("--conf-threshold", type=float, default=0.25, help="Soglia confidenza YOLO.")
parser.add_argument("--output-dir", type=str, default="output_tracking", help="Cartella output.")
parser.add_argument("--clip-min", type=int, default=-1500, help="Clip min (.h5).")
parser.add_argument("--clip-max", type=int, default=1500, help="Clip max (.h5).")

args = parser.parse_args()

# 2. RICERCA FILE
input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.h5')))
file_type = 'h5'
if not input_files:
    input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.jpg'))) + \
                  sorted(glob.glob(os.path.join(args.input_dir, '*.jpeg')))
    file_type = 'jpg'

if not input_files:
    print(f"ERRORE: Nessun file trovato in {args.input_dir}")
    exit()

print(f"Trovati {len(input_files)} file ({file_type}). Inizio elaborazione completa.")

# 3. SETUP OUTPUT E MODELLO
dir_norfair = os.path.join(args.output_dir, "norfair_only")
dir_labels = os.path.join(args.output_dir, "labels_only")
dir_combined = os.path.join(args.output_dir, "combined")

os.makedirs(dir_norfair, exist_ok=True)
os.makedirs(dir_labels, exist_ok=True)
os.makedirs(dir_combined, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Caricamento modello YOLOv7...")
model = attempt_load(args.model_path, map_location=device)
model.eval()
print("Modello caricato.")

tracker = Tracker(
    distance_function="iou",
    distance_threshold=0.7,
)

# 4. CICLO DI ELABORAZIONE (SU TUTTI I FILE)
for file_path in tqdm(input_files, desc="Tracking"):
    
    # --- CARICAMENTO IMMAGINE ---
    if file_path.endswith('.h5'):
        with h5py.File(file_path, 'r') as f:
            data = f['magnetogram/data'][:]
        background_mask = np.isnan(data)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        clipped_data = np.clip(data, args.clip_min, args.clip_max)
        normalized_data = (clipped_data - args.clip_min) / (args.clip_max - args.clip_min)
        resized_image = cv2.resize(normalized_data, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
        display_image_gray = (resized_image * 255).astype(np.uint8)
        background_mask_resized = cv2.resize(background_mask.astype(np.uint8), (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
        display_image_gray[background_mask_resized == 1] = 0
        base_image = cv2.cvtColor(display_image_gray, cv2.COLOR_GRAY2BGR)
        image_tensor_np = np.stack([resized_image] * 3, axis=-1)
    else:
        img_raw = cv2.imread(file_path)
        if img_raw is None: continue
        base_image = cv2.resize(img_raw, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
        image_tensor_np = base_image.astype(np.float32) / 255.0

    # Creazione delle 3 copie
    img_norfair_only = base_image.copy()
    img_labels_only = base_image.copy()
    img_combined = base_image.copy()

    # --- INFERENZA ---
    image_tensor = torch.from_numpy(image_tensor_np.transpose(2, 0, 1)).float().to(device)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        results = model(image_tensor, augment=False)[0]

    results = non_max_suppression(results, conf_thres=args.conf_threshold)[0]
    
    if results is not None and len(results) > 0:
        detections = yolo_detections_to_norfair_detections(results)
        tracked_objects = tracker.update(detections=detections)
    else:
        tracked_objects = tracker.update(detections=[])

    # --- DISEGNO GROUND TRUTH (ROSSO) ---
    label_path = get_label_path_from_image(file_path, args.labels_dir)
    gt_boxes = load_ground_truth(label_path, args.img_size) 
    
    for box in gt_boxes:
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        # Labels Only (Solo rettangolo, niente testo GT)
        cv2.rectangle(img_labels_only, pt1, pt2, (0, 0, 255), 2)
        
        # Combined (Solo rettangolo, niente testo GT)
        cv2.rectangle(img_combined, pt1, pt2, (0, 0, 255), 2)

    # --- DISEGNO TRACKING (NORFAIR - BLU) ---
    for obj in tracked_objects:
        if obj.estimate is None: continue
        
        points = obj.estimate.astype(int)
        pt1 = tuple(points[0])
        pt2 = tuple(points[1])
        
        # MODIFICA: Solo il numero ID
        label_text = str(obj.id)
        
        # Norfair Only
        cv2.rectangle(img_norfair_only, pt1, pt2, (255, 0, 0), 2)
        cv2.putText(img_norfair_only, label_text, (pt1[0], pt1[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Combined
        cv2.rectangle(img_combined, pt1, pt2, (255, 0, 0), 2)
        cv2.putText(img_combined, label_text, (pt1[0], pt1[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # --- SALVATAGGIO ---
    output_filename = os.path.basename(file_path)
    if output_filename.endswith('.h5'):
        output_filename = output_filename.replace('.h5', '.jpg')
    
    cv2.imwrite(os.path.join(dir_norfair, output_filename), img_norfair_only)
    cv2.imwrite(os.path.join(dir_labels, output_filename), img_labels_only)
    cv2.imwrite(os.path.join(dir_combined, output_filename), img_combined)

print("\nGenerazione immagini completata.")

# BLOCCO CREAZIONE VIDEO
if len(input_files) > 0:
    print("\nCreazione dei video (x3)...")
    framerate = 10 
    video_configs = [
        (dir_norfair, "video_norfair.mp4"),
        (dir_labels, "video_labels.mp4"),
        (dir_combined, "video_combined.mp4")
    ]
    for src_dir, video_name in video_configs:
        video_path = os.path.join(args.output_dir, video_name)
        print(f"Generazione {video_name}...")
        command = [
            'ffmpeg', '-y', '-r', str(framerate),
            '-pattern_type', 'glob', '-i', f'{src_dir}/*.jpg', 
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_path
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f" -> Creato: {video_path}")
        except Exception as e:
            print(f" -> Errore: {e}")
else:
    print("Nessun video generato (nessun file input).")

print("\nProcesso terminato con successo.")