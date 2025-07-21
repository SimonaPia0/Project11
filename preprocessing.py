import os
import json
from PIL import Image, ImageDraw
from tqdm import tqdm

# Classi normali (tutto il resto sarà considerato anomalia)
NORMAL_CLASSES = {'free', 'road', 'unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'background'}

def process_split(split, root_dir, out_root):
    print(f'\n🔄 Processing split: {split}')
    img_dir = os.path.join(root_dir, 'leftImg8bit', split)
    ann_dir = os.path.join(root_dir, 'gtCoarse', split)

    out_img_dir = os.path.join(out_root, split, 'images')
    out_mask_dir = os.path.join(out_root, split, 'masks')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    for root, _, files in tqdm(os.walk(img_dir)):
        for file in files:
            if not file.endswith('_leftImg8bit.png'):
                continue

            base = file.replace('_leftImg8bit.png', '')
            rel_dir = os.path.relpath(root, img_dir)
            img_path = os.path.join(root, file)
            json_path = os.path.join(ann_dir, rel_dir, f'{base}_gtCoarse_polygons.json')

            if not os.path.exists(json_path):
                continue

            # Load image and annotation
            image = Image.open(img_path).convert('RGB')
            with open(json_path, 'r') as f:
                ann_data = json.load(f)

            mask = Image.new('L', image.size, 255)
            draw = ImageDraw.Draw(mask)

            for obj in ann_data['objects']:
                label = obj['label']
                is_anomaly = label not in NORMAL_CLASSES
                fill_value = 1 if is_anomaly else 0
                polygon = [tuple(point) for point in obj['polygon']]
                draw.polygon(polygon, fill=fill_value)

            # Output name includes relative folder if needed
            out_name = f"{rel_dir.replace(os.sep, '_')}_{base}.png"
            image.save(os.path.join(out_img_dir, out_name))
            mask.save(os.path.join(out_mask_dir, out_name))

def main():
    ROOT = 'C:/Users/pinto/Downloads/dataset_LF'   # <-- Cambia con il path corretto
    OUT_ROOT = './dataset/lostandfound'     # <-- Cambia con il path di output

    for split in ['train', 'test']:
        process_split(split, ROOT, OUT_ROOT)

if __name__ == '__main__':
    main()