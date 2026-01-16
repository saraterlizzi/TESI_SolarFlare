# File: track.py

import argparse
import os
from typing import List
import subprocess
from datetime import datetime

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
def extract_datetime_from_filename(filename):
    """Estrae la data e l'ora dal nome del file per il filtro temporale."""
    base = os.path.basename(filename)
    name_no_ext = os.path.splitext(base)[0]
    try:
        # Formato standard ..._YYYY-MM-DD_HH-MM-SS
        date_str = name_no_ext[-19:] 
        dt_obj = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
        return dt_obj
    except ValueError:
        # Fallback per formati leggermente diversi
        parts = name_no_ext.split('_')
        if len(parts) >= 2:
            try:
                date_str = f"{parts[-2]}_{parts[-1]}"
                dt_obj = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
                return dt_obj
            except ValueError:
                return None
        return None

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

def get_color_by_id(id_obj):
    """
    Restituisce un colore (B, G, R) consistente basato sull'ID dell'oggetto.
    """
    palette = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (255, 128, 0),  # Deep Blue
        (0, 128, 255),  # Orange
        (128, 0, 255),  # Pinkish
        (255, 0, 128),  # Violet
        (128, 255, 0),  # Lime
        (0, 255, 128),  # Spring Green
    ]
    return palette[id_obj % len(palette)]

# --- INIZIO SCRIPT PRINCIPALE ---
# 1. DEFINIZIONE DEGLI ARGOMENTI
parser = argparse.ArgumentParser(description="Traccia le regioni attive solari (Struttura output organizzata).")
parser.add_argument("--input-dir", type=str, required=True, help="Cartella input (.h5 o .jpg).")
parser.add_argument("--model-path", type=str, required=True, help="Modello .pt.")
parser.add_argument("--img-size", type=int, default=640, help="Dimensione immagine.")
parser.add_argument("--conf-threshold", type=float, default=0.25, help="Soglia confidenza YOLO.")
parser.add_argument("--output-dir", type=str, default="output_tracking", help="Cartella output principale.")
parser.add_argument("--clip-min", type=int, default=-1500, help="Clip min (.h5).")
parser.add_argument("--clip-max", type=int, default=1500, help="Clip max (.h5).")
# Filtri temporali
parser.add_argument("--start-date", type=str, default=None, help="Inizio filtro (es. 2012-05-14_01-36-00)")
parser.add_argument("--end-date", type=str, default=None, help="Fine filtro (es. 2012-05-18_00-00-00)")

args = parser.parse_args()

# 2. RICERCA FILE E FILTRO TEMPORALE
input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.h5')))
file_type = 'h5'
if not input_files:
    input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.jpg'))) + \
                  sorted(glob.glob(os.path.join(args.input_dir, '*.jpeg')))
    file_type = 'jpg'

if not input_files:
    print(f"ERRORE: Nessun file trovato in {args.input_dir}")
    exit()

# Filtro
files_to_process = []
if args.start_date or args.end_date:
    print("\nApplicazione filtro temporale...")
    start_dt = datetime.min
    end_dt = datetime.max
    
    if args.start_date:
        try:
            start_dt = datetime.strptime(args.start_date, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            print("ERRORE DATA START. Usa formato YYYY-MM-DD_HH-MM-SS")
            exit()
    if args.end_date:
        try:
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            print("ERRORE DATA END. Usa formato YYYY-MM-DD_HH-MM-SS")
            exit()

    for f in input_files:
        dt = extract_datetime_from_filename(f)
        if dt and start_dt <= dt <= end_dt:
            files_to_process.append(f)
            
    if not files_to_process:
        print("Nessun file nell'intervallo temporale.")
        exit()
    print(f"File selezionati: {len(files_to_process)} (su {len(input_files)})")
else:
    files_to_process = input_files
    print(f"File da elaborare: {len(files_to_process)}")

# 3. SETUP OUTPUT (CARTELLA FRAME + CARTELLA VIDEO)
# Creazione sottocartella per i frame
frames_dir = os.path.join(args.output_dir, "frames")
os.makedirs(frames_dir, exist_ok=True)
video_output_path = os.path.join(args.output_dir, "tracking_video.mp4")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("\nCaricamento modello YOLOv7...")
model = attempt_load(args.model_path, map_location=device)
model.eval()

tracker = Tracker(
    distance_function="iou",
    distance_threshold=0.7,
)

# 4. CICLO DI ELABORAZIONE
for file_path in tqdm(files_to_process, desc="Tracking"):
    
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
        display_image = cv2.cvtColor(display_image_gray, cv2.COLOR_GRAY2BGR)
        image_tensor_np = np.stack([resized_image] * 3, axis=-1)
    else:
        img_raw = cv2.imread(file_path)
        if img_raw is None: continue
        display_image = cv2.resize(img_raw, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
        image_tensor_np = display_image.astype(np.float32) / 255.0

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

    # --- DISEGNO TRACKING (SOLO BOX, COLORI DIVERSI, NO TESTO) ---
    for obj in tracked_objects:
        if obj.estimate is None: continue
        
        points = obj.estimate.astype(int)
        pt1 = tuple(points[0])
        pt2 = tuple(points[1])
        
        # Colore Dinamico
        color = get_color_by_id(obj.id)
        
        # Solo rettangolo colorato
        cv2.rectangle(display_image, pt1, pt2, color, 2)

    # --- SALVATAGGIO NELLA SOTTOCARTELLA 'FRAMES' ---
    output_filename = os.path.basename(file_path)
    if output_filename.endswith('.h5'):
        output_filename = output_filename.replace('.h5', '.jpg')
    
    cv2.imwrite(os.path.join(frames_dir, output_filename), display_image)

print(f"\nGenerazione frame completata. Salvati in: {frames_dir}")

# BLOCCO CREAZIONE VIDEO (NELLA CARTELLA PARENT)
if len(files_to_process) > 0:
    print("\nCreazione video...")
    framerate = 10 
    
    command = [
        'ffmpeg', '-y', '-r', str(framerate),
        '-pattern_type', 'glob', '-i', f'{frames_dir}/*.jpg', 
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f" -> Video creato con successo: {video_output_path}")
    except Exception as e:
        print(f" -> Errore FFmpeg: {e}")
else:
    print("Nessun frame per creare il video.")

print("\nFinito.")