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

from models.experimental import attempt_load
from utils.general import non_max_suppression

def center(points):
    return [np.mean(np.array(points), axis=0)]

def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor,
) -> List[Detection]:
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

parser = argparse.ArgumentParser(description="Traccia le regioni attive solari su una sequenza di magnetogrammi.")
parser.add_argument("--input-dir", type=str, required=True, help="Percorso alla cartella contenente i file .h5 in sequenza.")
parser.add_argument("--model-path", type=str, required=True, help="Percorso al file .pt del tuo modello addestrato (es. best.pt).")
parser.add_argument("--img-size", type=int, default=640, help="Dimensione a cui ridimensionare le immagini per l'inferenza.")
parser.add_argument("--conf-threshold", type=float, default=0.25, help="Soglia di confidenza per le rilevazioni.")
parser.add_argument("--output-dir", type=str, default="output_tracking", help="Cartella dove salvare le immagini con il tracking.")
parser.add_argument("--clip-min", type=int, default=-1500, help="Valore minimo per il clipping.")
parser.add_argument("--clip-max", type=int, default=1500, help="Valore massimo per il clipping.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Caricamento del modello...")
model = attempt_load(args.model_path, map_location=device)
model.eval()
print("Modello caricato.")

tracker = Tracker(
    distance_function="iou",
    distance_threshold=0.7,
)

h5_files = sorted(glob.glob(os.path.join(args.input_dir, '*.h5')))
if not h5_files:
    print(f"ERRORE: Nessun file .h5 trovato nella cartella {args.input_dir}")
    exit()

for h5_path in tqdm(h5_files, desc="Tracking delle regioni attive"):
    with h5py.File(h5_path, 'r') as f:
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
    
    image_tensor = np.stack([resized_image] * 3, axis=-1)
    image_tensor = torch.from_numpy(image_tensor.transpose(2, 0, 1)).float().to(device)
    image_tensor = image_tensor.unsqueeze(0)  

    with torch.no_grad():
        results = model(image_tensor, augment=False)[0]

    results = non_max_suppression(results, conf_thres=args.conf_threshold)[0]
    
    if results is not None and len(results) > 0:
        detections = yolo_detections_to_norfair_detections(results)
        
        tracked_objects = tracker.update(detections=detections)
    else:
        tracked_objects = tracker.update(detections=[])

    norfair.draw_tracked_boxes(display_image, tracked_objects)

    output_filename = os.path.basename(h5_path) + ".jpg"
    cv2.imwrite(os.path.join(args.output_dir, output_filename), display_image)

print(f"\nProcesso completato. Le immagini con il tracciamento sono state salvate in '{args.output_dir}'.")


print("\nCreazione del video in corso con FFmpeg...")
output_dir = args.output_dir
video_path = os.path.join(output_dir, "tracking_video.mp4")
framerate = 10  

command = [
    'ffmpeg',
    '-y', 
    '-r', str(framerate),
    '-pattern_type', 'glob',
    '-i', f'{output_dir}/*.jpg',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    video_path
]

try:
    subprocess.run(command, check=True)
    print(f"Video creato con successo: {video_path}")
except FileNotFoundError:
    print("\nERRORE: FFmpeg non e' installato o non e' nel PATH. Impossibile creare il video.")
    print("Puoi creare il video manualmente eseguendo questo comando dalla cartella del progetto:")
    print(f"ffmpeg -r {framerate} -pattern_type glob -i '{output_dir}/*.jpg' -c:v libx264 -pix_fmt yuv420p {video_path}")
except subprocess.CalledProcessError as e:
    print(f"\nERRORE durante la creazione del video con FFmpeg: {e}")