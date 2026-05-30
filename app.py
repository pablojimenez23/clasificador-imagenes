import torch
from torchvision import transforms, models
from PIL import Image
import sys
import os

# Clases en orden alfabetico, igual que las lee ImageFolder
clases = [
    'battery', 'biological', 'brown-glass', 'cardboard',
    'clothes', 'green-glass', 'metal', 'paper',
    'plastic', 'shoes', 'trash', 'white-glass'
]

def cargar_modelo():
    # Carga el modelo con la arquitectura y los pesos entrenados
    modelo = models.resnet18(weights=None)
    modelo.fc = torch.nn.Linear(modelo.fc.in_features, len(clases))
    modelo.load_state_dict(
        torch.load('modelo/clasificador.pth', map_location='cpu'))
    modelo.eval()
    return modelo

def preprocesar(ruta):
    # Aplica las mismas transformaciones usadas en validacion
    transformacion = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    imagen = Image.open(ruta).convert('RGB')
    return transformacion(imagen).unsqueeze(0)

def predecir(ruta):
    if not os.path.exists(ruta):
        print(f'No se encontro la imagen: {ruta}')
        return

    modelo = cargar_modelo()
    tensor = preprocesar(ruta)

    with torch.no_grad():
        salida         = modelo(tensor)
        probabilidades = torch.softmax(salida, dim=1)
        confianza, indice = probabilidades.max(1)

    print(f'\nImagen:        {ruta}')
    print(f'Clasificacion: {clases[indice.item()].upper()}')
    print(f'Confianza:     {confianza.item()*100:.1f}%')

    # Mostrar las 3 predicciones con mayor probabilidad
    print('\nTop 3:')
    valores, indices = probabilidades[0].topk(3)
    for prob, idx in zip(valores, indices):
        print(f'  {clases[idx]:15s} -> {prob*100:.1f}%')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Uso: python app.py ruta/imagen.jpg')
    else:
        predecir(sys.argv[1])