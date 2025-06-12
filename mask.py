import numpy as np
from PIL import Image

mask = np.array(Image.open('dataset/train/label/train10.png'))
print(np.unique(mask))

import numpy as np
from PIL import Image
import glob

mask_files = glob.glob("dataset/train/label/*.png")  # o .png/.jpg ecc.

all_unique_labels = set()

for mask_file in mask_files:
    mask = np.array(Image.open(mask_file))
    unique_labels = np.unique(mask)
    all_unique_labels.update(unique_labels)

print("Tutte le classi trovate nel dataset:", sorted(all_unique_labels))
