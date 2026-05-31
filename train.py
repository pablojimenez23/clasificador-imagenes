import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Configuracion general del entrenamiento
dispositivo  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epocas       = 10
tam_lote     = 32
tasa_aprend  = 0.001
clases       = [
    'battery', 'biological', 'brown-glass', 'cardboard',
    'clothes', 'green-glass', 'metal', 'paper',
    'plastic', 'shoes', 'trash', 'unknown', 'white-glass'
]

print(f'Dispositivo: {dispositivo}')
print(f'Total de clases: {len(clases)}')

# Transformaciones para entrenamiento con aumento de datos
transform_entren = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transformaciones para validacion sin aumento de datos
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carga de imagenes desde carpetas organizadas por clase
conjunto_entren = datasets.ImageFolder('data/train', transform=transform_entren)
conjunto_val    = datasets.ImageFolder('data/val',   transform=transform_val)

cargador_entren = torch.utils.data.DataLoader(
    conjunto_entren, batch_size=tam_lote, shuffle=True, num_workers=2)
cargador_val = torch.utils.data.DataLoader(
    conjunto_val, batch_size=tam_lote, shuffle=False)

print(f'Imagenes de entrenamiento: {len(conjunto_entren)}')
print(f'Imagenes de validacion:    {len(conjunto_val)}')
print(f'Clases detectadas: {conjunto_entren.classes}')

# Modelo ResNet18 preentrenado en ImageNet con Transfer Learning
modelo = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Se congelan todas las capas para no modificar lo aprendido previamente
for parametro in modelo.parameters():
    parametro.requires_grad = False

# Se reemplaza la ultima capa para adaptarla a las clases del proyecto
num_caracteristicas = modelo.fc.in_features
modelo.fc = nn.Linear(num_caracteristicas, len(clases))
modelo     = modelo.to(dispositivo)

# Funcion de perdida y optimizador aplicados solo a la ultima capa
criterio    = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.fc.parameters(), lr=tasa_aprend)

# Reductor de tasa de aprendizaje si el modelo deja de mejorar
planificador = optim.lr_scheduler.ReduceLROnPlateau(
    optimizador, patience=2, factor=0.5)

historial_entren = []
historial_val    = []

for epoca in range(epocas):

    # Fase de entrenamiento
    modelo.train()
    total, correctos = 0, 0

    for imagenes, etiquetas in cargador_entren:
        imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
        optimizador.zero_grad()
        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)
        perdida.backward()
        optimizador.step()
        _, prediccion = salidas.max(1)
        total    += etiquetas.size(0)
        correctos += prediccion.eq(etiquetas).sum().item()

    precision_entren = 100 * correctos / total
    historial_entren.append(precision_entren)

    # Fase de validacion sin calculo de gradientes
    modelo.eval()
    total_val, correctos_val = 0, 0

    with torch.no_grad():
        for imagenes, etiquetas in cargador_val:
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
            salidas = modelo(imagenes)
            _, prediccion = salidas.max(1)
            total_val    += etiquetas.size(0)
            correctos_val += prediccion.eq(etiquetas).sum().item()

    precision_val = 100 * correctos_val / total_val
    historial_val.append(precision_val)

    planificador.step(100 - precision_val)
    print(f'Epoca {epoca+1:02d}/{epocas} | Entrenamiento: {precision_entren:.1f}% | Validacion: {precision_val:.1f}%')

# Guardar los pesos del modelo entrenado
os.makedirs('modelo', exist_ok=True)
torch.save(modelo.state_dict(), 'modelo/clasificador.pth')
print('\nModelo guardado en modelo/clasificador.pth')

# Generar y guardar grafica de precision por epoca
plt.figure(figsize=(8, 5))
plt.plot(range(1, epocas+1), historial_entren, label='Entrenamiento', marker='o')
plt.plot(range(1, epocas+1), historial_val,    label='Validacion',    marker='s')
plt.xlabel('Epoca')
plt.ylabel('Precision (%)')
plt.title('Curva de entrenamiento')
plt.legend()
plt.grid(True)
plt.savefig('modelo/curva_entrenamiento.png', dpi=150)
print('Curva guardada en modelo/curva_entrenamiento.png')