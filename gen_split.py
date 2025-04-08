import os
import shutil
import random

img_dir = 'shapes_dataset/images'
label_dir = 'shapes_dataset/labels'

output_img_dir = {
    'train': 'shapes_dataset/images/train',
    'val': 'shapes_dataset/images/val',
    'test': 'shapes_dataset/images/test'
}

output_lbl_dir = {
    'train': 'shapes_dataset/labels/train',
    'val': 'shapes_dataset/labels/val',
    'test': 'shapes_dataset/labels/test'
}

for d in output_img_dir.values():
    os.makedirs(d, exist_ok=True)
for d in output_lbl_dir.values():
    os.makedirs(d, exist_ok=True)

# ✅ Only pick image files, not folders
images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
random.shuffle(images)

train_split = int(0.7 * len(images))
val_split = int(0.85 * len(images))

splits = {
    'train': images[:train_split],
    'val': images[train_split:val_split],
    'test': images[val_split:]
}

for split, files in splits.items():
    for file in files:
        name = os.path.splitext(file)[0]
        shutil.copy(f"{img_dir}/{file}", f"{output_img_dir[split]}/{file}")
        shutil.copy(f"{label_dir}/{name}.txt", f"{output_lbl_dir[split]}/{name}.txt")

print("Dataset split into train/val/test.")