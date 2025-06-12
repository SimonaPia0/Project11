import os
import numpy as np
from PIL import Image
import json

# === CONFIG ===
dataset_root = './dataset'
splits = ['train', 'val']
label_map_path = os.path.join(dataset_root, 'label_map.json')

# === 1. Trova tutte le etichette uniche da train e val ===
all_labels = set()
for split in splits:
    label_dir = os.path.join(dataset_root, split, 'label')
    for fname in os.listdir(label_dir):
        path = os.path.join(label_dir, fname)
        mask = np.array(Image.open(path).convert("L"))
        all_labels.update(np.unique(mask))

all_labels = sorted(list(all_labels))
label_map = {val: idx for idx, val in enumerate(all_labels)}

print(f"Etichette trovate: {all_labels}")
print(f"Label map creata con {len(label_map)} classi.")

# === 2. Salva la label_map in un JSON ===
with open(label_map_path, 'w') as f:
    json.dump({str(k): v for k, v in label_map.items()}, f)
print(f"Label map salvata in: {label_map_path}")

# === 3. Rimappa le label per ogni split ===
for split in splits:
    print(f"\nRimappatura {split}...")
    label_dir = os.path.join(dataset_root, split, 'label')
    output_dir = os.path.join(dataset_root, split, 'label_remapped')
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(label_dir):
        in_path = os.path.join(label_dir, fname)
        out_path = os.path.join(output_dir, fname)

        mask = np.array(Image.open(in_path).convert("L"))
        remapped = np.vectorize(label_map.get)(mask).astype(np.uint8)
        Image.fromarray(remapped).save(out_path)

    print(f"Salvato in: {output_dir}")