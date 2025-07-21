import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import namedtuple

# Definizione delle classi Cityscapes con train_id e colori
CityscapesClass = namedtuple('CityscapesClass', [
    'name', 'id', 'train_id', 'category', 'category_id',
    'has_instances', 'ignore_in_eval', 'color'
])

classes = [
    CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road',                 7, 0,   'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk',             8, 1,   'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track',           10,255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building',             11, 2,  'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall',                 12, 3,  'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence',                13, 4,  'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail',           14,255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge',               15,255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel',               16,255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole',                 17, 5,  'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup',            18,255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light',        19, 6,  'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign',         20, 7,  'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation',           21, 8,  'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain',              22, 9,  'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky',                  23,10,  'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person',               24,11,  'human', 6, True,  False, (220, 20, 60)),
    CityscapesClass('rider',                25,12,  'human', 6, True,  False, (255, 0, 0)),
    CityscapesClass('car',                  26,13,  'vehicle', 7, True,  False, (0, 0, 142)),
    CityscapesClass('truck',                27,14,  'vehicle', 7, True,  False, (0, 0, 70)),
    CityscapesClass('bus',                  28,15,  'vehicle', 7, True,  False, (0, 60, 100)),
    CityscapesClass('caravan',              29,255, 'vehicle', 7, True,  True,  (0, 0, 90)),
    CityscapesClass('trailer',              30,255, 'vehicle', 7, True,  True,  (0, 0, 110)),
    CityscapesClass('train',                31,16,  'vehicle', 7, True,  False, (0, 80, 100)),
    CityscapesClass('motorcycle',           32,17,  'vehicle', 7, True,  False, (0, 0, 230)),
    CityscapesClass('bicycle',              33,18,  'vehicle', 7, True,  False, (119, 11, 32)),
    CityscapesClass('license plate',        -1,255, 'vehicle', 7, False, True, (0, 0, 142)),
    CityscapesClass('unknown',              255,20,  'void', 0, False, False, (255, 0, 0)),
]

train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color.append([0, 0, 0])  # colore per ignore
train_id_to_color = np.array(train_id_to_color)

id_to_train_id = np.array([c.train_id for c in classes])

# Mappa da id originali a train_id (256 valori)
full_id_to_train_id = np.ones(256, dtype=np.uint8) * 255
for cls_ in classes:
    if 0 <= cls_.id < 256:
        full_id_to_train_id[cls_.id] = cls_.train_id

def encode_target(mask):
    """Converti maschera con id originali in maschera con train_id."""
    return full_id_to_train_id[mask]

def decode_target(train_id_mask):
    """Converti maschera train_id in maschera RGB colore."""
    mask = train_id_mask.copy()
    mask[mask == 255] = len(train_id_to_color) - 1  # ignora -> nero
    return train_id_to_color[mask]

class LostAndFoundDatasetFromMasks(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.img_dir = os.path.join(root, split, 'images')
        self.mask_dir = os.path.join(root, split, 'masks')

        # Carica tutti i file immagine (assumendo .png o .jpg)
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.img_dir, img_name)

        # Supponiamo che la maschera abbia lo stesso nome con _mask
        mask_name = img_name
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # scala di grigi

        mask_np = np.array(mask)
        mask_train_id = encode_target(mask_np)

        # Se usi trasformazioni, applicale all'immagine e alla maschera train_id
        if self.transform:
            image, mask_train_id = self.transform(image, mask_train_id)

        # Converti mask_train_id in tensor prima di restituirlo
        mask_train_id = torch.from_numpy(mask_train_id).long()

        return image, mask_train_id