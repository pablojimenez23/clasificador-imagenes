# Guia de instalacion y uso

---

## Requisitos

Python: 3.10 o superior
pip: incluido con Python
Git: https://git-scm.com/downloads
Cuenta en Kaggle: https://www.kaggle.com

---

## 1. Clonar el repositorio

Clona el proyecto con: `git clone https://github.com/pablojimenez23/clasificador-imagenes.git`

Entra a la carpeta con: `cd clasificador-imagenes`

---

## 2. Instalar dependencias

Instala todas las librerias con: `pip install -r requirements.txt`

---

## 3. Configurar Kaggle

Crea una cuenta en https://www.kaggle.com y genera un token en Settings - API - Create New Token.

Guarda el archivo kaggle.json en:

Windows: `C:\Users\TuUsuario\.kaggle\kaggle.json`
Mac/Linux: `~/.kaggle/kaggle.json`

El contenido del archivo debe ser:

```json
{
  "username": "tu_usuario_kaggle",
  "key": "tu_token_kaggle"
}
```

---

## 4. Descargar el dataset

Instala la herramienta de Kaggle con: `pip install kaggle`

Descarga el dataset con: `kaggle datasets download -d mostafaabla/garbage-classification`

Descomprime con: `unzip garbage-classification.zip -d data_raw`

---

## 5. Organizar el dataset

Distribuye las imagenes en carpetas de entrenamiento y validacion con: `python organizar_datos.py`

Salida esperada:

```
battery        : 756 entrenamiento | 189 validacion
biological     : 788 entrenamiento | 197 validacion
brown-glass    : 485 entrenamiento | 122 validacion
cardboard      : 712 entrenamiento | 179 validacion
clothes        : 4260 entrenamiento | 1065 validacion
green-glass    : 503 entrenamiento | 126 validacion
metal          : 615 entrenamiento | 154 validacion
paper          : 840 entrenamiento | 210 validacion
plastic        : 692 entrenamiento | 173 validacion
shoes          : 1581 entrenamiento | 396 validacion
trash          : 557 entrenamiento | 140 validacion
white-glass    : 620 entrenamiento | 155 validacion

Dataset organizado correctamente
```

---

## 6. Entrenar el modelo

Con CPU localmente: `python train.py`

Con GPU en Google Colab (recomendado, aproximadamente 10 minutos):

Abre https://colab.research.google.com, activa GPU en Entorno de ejecucion - Cambiar tipo de entorno - T4 GPU, sube train.py y organizar_datos.py junto con la carpeta data/ y ejecuta:

```bash
!python organizar_datos.py
!python train.py
```

Salida esperada:

```
Dispositivo: cuda
Total de clases: 12
Imagenes de entrenamiento: 12409
Imagenes de validacion:    3106
Epoca 01/10 | Entrenamiento: 78.6% | Validacion: 88.9%
Epoca 02/10 | Entrenamiento: 87.6% | Validacion: 91.3%
Epoca 10/10 | Entrenamiento: 90.9% | Validacion: 91.7%

Modelo guardado en modelo/clasificador.pth
Curva guardada en modelo/curva_entrenamiento.png
```

Descarga modelo/clasificador.pth y guardalo en la carpeta modelo/ de tu proyecto local.

---

## 7. Clasificar una imagen

Clasifica cualquier imagen con: `python app.py ruta/a/imagen.jpg`

Ejemplo: `python app.py data/val/plastic/plastic10.jpg`

Salida esperada:

```
Imagen:        data/val/plastic/plastic10.jpg
Clasificacion: PLASTIC
Confianza:     90.3%

Top 3:
  plastic         -> 90.3%
  white-glass     -> 8.5%
  metal           -> 1.1%
```

---

## Solucion de problemas

Error data/train no encontrado: El dataset no fue organizado. Ejecuta primero: `python organizar_datos.py`

Error clasificador.pth no encontrado: El modelo no fue entrenado o no esta en la carpeta correcta. Ejecuta: `python train.py` o descargalo desde Colab y colocalo en modelo/

Error kaggle.json no encontrado: Verifica que el archivo este en la ruta correcta segun tu sistema operativo.

---

Para mas informacion ver [README.md](README.md)