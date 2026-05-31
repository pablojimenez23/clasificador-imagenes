import os
import shutil
import random

origen = 'data_raw_unknown/seg_train/seg_train'
clases = os.listdir(origen)
random.seed(42)

todas = []
for clase in clases:
    carpeta = os.path.join(origen, clase)
    imgs = [os.path.join(carpeta, f) for f in os.listdir(carpeta)]
    todas.extend(imgs)

random.shuffle(todas)
corte = int(len(todas) * 0.8)
train = todas[:corte]
val   = todas[corte:]

os.makedirs('data/train/unknown', exist_ok=True)
os.makedirs('data/val/unknown',   exist_ok=True)

for img in train:
    shutil.copy(img, os.path.join('data/train/unknown', os.path.basename(img)))
for img in val:
    shutil.copy(img, os.path.join('data/val/unknown', os.path.basename(img)))

print(f'unknown: {len(train)} entrenamiento | {len(val)} validacion')