import os
import shutil
import random

# Ruta donde se descomprimio el dataset descargado
ORIGEN = 'data_raw/garbage_classification'

# Lista de clases del dataset
CLASES = [
    'battery', 'biological', 'brown-glass', 'cardboard',
    'clothes', 'green-glass', 'metal', 'paper',
    'plastic', 'shoes', 'trash', 'white-glass'
]

# Porcentaje de imagenes destinadas a validacion
PORCENTAJE_VAL = 0.2

random.seed(42)

for clase in CLASES:
    # Crear carpetas de destino si no existen
    os.makedirs(f'data/train/{clase}', exist_ok=True)
    os.makedirs(f'data/val/{clase}',   exist_ok=True)

    carpeta_origen = os.path.join(ORIGEN, clase)
    imagenes = os.listdir(carpeta_origen)
    random.shuffle(imagenes)

    # Dividir imagenes en entrenamiento y validacion
    corte            = int(len(imagenes) * (1 - PORCENTAJE_VAL))
    imagenes_entren  = imagenes[:corte]
    imagenes_val     = imagenes[corte:]

    for imagen in imagenes_entren:
        shutil.copy(
            os.path.join(carpeta_origen, imagen),
            os.path.join('data/train', clase, imagen)
        )

    for imagen in imagenes_val:
        shutil.copy(
            os.path.join(carpeta_origen, imagen),
            os.path.join('data/val', clase, imagen)
        )

    print(f'{clase:15s}: {len(imagenes_entren)} entrenamiento | {len(imagenes_val)} validacion')

print('\nDataset organizado correctamente')