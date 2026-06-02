import os
import shutil
import random

# Ruta donde se descomprimio el dataset de animales
origen = 'data_raw_animals/raw-img'
clases = os.listdir(origen)
random.seed(42)

# Recopilar todas las imagenes de todas las categorias
todas = []
for clase in clases:
    carpeta = os.path.join(origen, clase)
    if os.path.isdir(carpeta):
        imgs = [os.path.join(carpeta, f) for f in os.listdir(carpeta)]
        todas.extend(imgs)

random.shuffle(todas)

# Usar solo 3000 imagenes para no desbalancear el dataset
todas = todas[:3000]

# Dividir en entrenamiento y validacion
corte = int(len(todas) * 0.8)
train = todas[:corte]
val   = todas[corte:]

for img in train:
    shutil.copy(img, os.path.join('data/train/unknown', os.path.basename(img)))

for img in val:
    shutil.copy(img, os.path.join('data/val/unknown', os.path.basename(img)))

print(f'animales agregados a unknown: {len(train)} entrenamiento | {len(val)} validacion')
print('\nDataset animales organizado correctamente')