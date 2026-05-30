<div align="right">
  <img src="https://img.shields.io/badge/PJ-Pablo%20Jim%C3%A9nez-black?style=for-the-badge" alt="PJ"/>
</div>

# Clasificador de Residuos con Transfer Learning

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat-square&logo=pytorch)
![ResNet18](https://img.shields.io/badge/Modelo-ResNet--18-green?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-91.7%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

Clasificador de imagenes entrenado con Transfer Learning sobre ResNet-18 preentrenado en ImageNet, aplicado a la clasificacion automatica de residuos en 12 categorias. El modelo alcanza un 91.7% de precision en validacion entrenando unicamente la ultima capa de la red neuronal.

---

## Problema que resuelve

La clasificacion manual de residuos es lenta, costosa y propensa a errores humanos. Este modelo automatiza la identificacion del tipo de residuo a partir de una imagen, con aplicacion directa en plantas de reciclaje, aplicaciones moviles de clasificacion o sistemas de vision industrial.

---

## Resultados

Precision entrenamiento: 90.9%
Precision validacion: 91.7%
Epocas entrenadas: 10
Imagenes de entrenamiento: 12,409
Imagenes de validacion: 3,106
Total de clases: 12

![Curva de entrenamiento](assets/curva_entrenamiento.png)

---

## Clases detectadas

battery: Pilas y baterias
biological: Residuos biologicos
brown-glass: Vidrio cafe
cardboard: Carton
clothes: Ropa y textiles
green-glass: Vidrio verde
metal: Metal
paper: Papel
plastic: Plastico
shoes: Calzado
trash: Basura general
white-glass: Vidrio blanco

---

## Tecnologias utilizadas

Python 3.12: Lenguaje principal — https://python.org
PyTorch 2.0: Framework de deep learning — https://pytorch.org
TorchVision: Modelos preentrenados y transformaciones — https://pytorch.org/vision
ResNet-18: Arquitectura de red neuronal convolucional — https://arxiv.org/abs/1512.03385
Matplotlib: Generacion de graficas de entrenamiento — https://matplotlib.org
Pillow: Procesamiento y apertura de imagenes — https://python-pillow.org
Kaggle: Fuente del dataset — https://kaggle.com
Google Colab: Entrenamiento con GPU T4 gratuita — https://colab.research.google.com

---

## Estructura del proyecto

```
clasificador-imagenes/
├── assets/
│   └── curva_entrenamiento.png   # Grafica de precision por epoca
├── data/                         # No incluido en el repositorio
│   ├── train/                    # Imagenes de entrenamiento por clase (80%)
│   └── val/                      # Imagenes de validacion por clase (20%)
├── modelo/                       # No incluido en el repositorio
│   └── clasificador.pth          # Pesos del modelo entrenado
├── train.py                      # Script de entrenamiento
├── app.py                        # Script de inferencia
├── organizar_datos.py            # Script de organizacion del dataset
├── requirements.txt              # Dependencias del proyecto
├── INSTALL.md                    # Guia de instalacion y uso
└── README.md
```

---

## Ejemplos de uso

Una vez instalado el proyecto (ver [INSTALL.md](INSTALL.md)), se puede clasificar cualquier imagen con: `python app.py ruta/a/imagen.jpg`

### Plastico

Comando: `python app.py data/val/plastic/plastic10.jpg`

```
Imagen:        data/val/plastic/plastic10.jpg
Clasificacion: PLASTIC
Confianza:     90.3%

Top 3:
  plastic         -> 90.3%
  white-glass     -> 8.5%
  metal           -> 1.1%
```

### Carton

Comando: `python app.py data/val/cardboard/cardboard10.jpg`

```
Imagen:        data/val/cardboard/cardboard10.jpg
Clasificacion: CARDBOARD
Confianza:     97.1%

Top 3:
  cardboard       -> 97.1%
  paper           -> 2.4%
  trash           -> 0.3%
```

### Vidrio verde

Comando: `python app.py data/val/green-glass/green-glass10.jpg`

```
Imagen:        data/val/green-glass/green-glass10.jpg
Clasificacion: GREEN-GLASS
Confianza:     94.8%

Top 3:
  green-glass     -> 94.8%
  brown-glass     -> 3.9%
  white-glass     -> 1.1%
```

### Ropa

Comando: `python app.py data/val/clothes/clothes10.jpg`

```
Imagen:        data/val/clothes/clothes10.jpg
Clasificacion: CLOTHES
Confianza:     98.2%

Top 3:
  clothes         -> 98.2%
  shoes           -> 1.4%
  trash           -> 0.3%
```

### Bateria

Comando: `python app.py data/val/battery/battery10.jpg`

```
Imagen:        data/val/battery/battery10.jpg
Clasificacion: BATTERY
Confianza:     95.6%

Top 3:
  battery         -> 95.6%
  metal           -> 3.1%
  trash           -> 1.1%
```

### Imagen propia

Tambien es posible clasificar cualquier imagen JPG o PNG desde cualquier ruta con: `python app.py C:/Users/TuUsuario/Desktop/foto_botella.jpg`

---

## Decisiones de diseno

Arquitectura ResNet-18: Se eligio por su balance entre precision y velocidad de inferencia frente a arquitecturas mas pesadas como VGG-16 o ResNet-50. Para 12 clases con este volumen de datos es suficiente y mas eficiente en produccion.

Transfer Learning: Se congelaron todas las capas de la red y se reentrenó unicamente la capa final. Permite alcanzar alta precision con relativamente pocas imagenes por clase, aprovechando los patrones visuales que ResNet-18 aprendio de ImageNet.

Data augmentation: Se aplico flip horizontal, rotacion de hasta 15 grados y variaciones de brillo y contraste solo en entrenamiento. Mejora la generalizacion del modelo sin necesidad de recolectar mas datos.

Division 80/20: Garantiza una evaluacion representativa del rendimiento real del modelo ante imagenes no vistas.

Scheduler de tasa de aprendizaje: Se uso ReduceLROnPlateau para reducir automaticamente la tasa de aprendizaje cuando el modelo deja de mejorar, evitando oscilaciones y mejorando la convergencia en las ultimas epocas.

---

## Dataset

Nombre: Garbage Classification Dataset
Fuente: Kaggle — https://www.kaggle.com/datasets/mostafaabla/garbage-classification
Total de imagenes: 15,515
Clases: 12
Formato: JPG

---

Para instalar y reproducir el proyecto ver [INSTALL.md](INSTALL.md)

---

## Autor

Pablo Jimenez — Ingeniero en Informatica
GitHub: https://github.com/pablojimenez23
Kaggle: https://www.kaggle.com/pablandjf

---

*MIT License · 2026*