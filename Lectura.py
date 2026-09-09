import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Definir las clases en el mismo orden exacto que se detectaron al entrenar
class_names = [
    'Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd', 
    'Golden_Retriever', 'Labrador_Retriever', 'Poodle', 'Rottweiler', 'Yorkshire_Terrier'
]

# 2. Cargar la arquitectura base y adaptarla a las 10 clases
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# 3. Cargar los pesos guardados del entrenamiento local
model.load_state_dict(torch.load("modelo_pytorch_local.pth", map_location=torch.device('cpu')))
model.eval()  # Cambiar el modelo a modo de evaluación

# 4. Transformaciones que deben coincidir con las del entrenamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Función para clasificar una nueva imagen local
def predecir_imagen(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Añadir dimensión de lote (batch size = 1)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    clase_predicha = class_names[predicted.item()]
    certeza = confidence.item() * 100
    
    print(f"Imagen: {image_path}")
    print(f"Raza predicha: {clase_predicha} ({certeza:.2f}% de confianza)\n")

# --- PRUEBA CON UNA IMAGEN ---
# Reemplaza 'ruta/a/tu/imagen.jpg' por la ubicación de cualquier foto que quieras probar
# Ejemplo: predecir_imagen("dataset/Beagle/tu_foto.jpg")
predecir_imagen("dataset\test\Yorktest.jpg")