import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root, split, 'img')
        self.label_dir = os.path.join(root, split, 'label')

        self.images = sorted(os.listdir(self.image_dir))
        self.labels = sorted(os.listdir(self.label_dir))

        assert len(self.images) == len(self.labels), "Image and label count mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.labels[index])

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L") 

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label

    def decode_target(self, mask):
        # Opzionale: visualizzazione colori per classi
        return mask