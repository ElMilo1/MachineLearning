import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

if __name__ == '__main__':
    # Transformaciones estándar para redimensionar y normalizar las imágenes
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Carga automática usando la carpeta local 'dataset'
    data_dir = "dataset"
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # División estricta en 60% entrenamiento y 40% validación
    train_size = int(0.6 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # num_workers=0 evita conflictos de multiprocesamiento en Windows con datasets pequeños
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    class_names = full_dataset.classes
    print("Clases detectadas automáticamente:", class_names)
    print(f"Total de registros: {len(full_dataset)} | Entrenamiento (60%): {train_size} | Validación (40%): {val_size}")

    # Carga del modelo base (ResNet18) adaptado a las clases detectadas
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # Configuración del dispositivo local (CPU)
    device = torch.device("cpu")
    model = model.to(device)

    # Función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Bucle de entrenamiento local
    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Pérdida (Loss): {running_loss/len(train_loader):.4f}")

    # Guardar los pesos del modelo entrenado localmente
    torch.save(model.state_dict(), "modelo_pytorch_local.pth")
    print("Entrenamiento finalizado y modelo guardado como 'modelo_pytorch_local.pth'.")