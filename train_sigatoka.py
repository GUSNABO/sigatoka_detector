import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# BLOQUE 1: Configuración y Preprocesamiento
# ==========================================
def create_dataloaders(data_dir, batch_size=32):
    """
    Configura las transformaciones y carga el dataset usando ImageFolder.
    Divide los datos en 80% entrenamiento y 20% validación.
    Aplica Data Augmentation de forma exclusiva al conjunto de entrenamiento.
    """
    # Transformaciones avanzadas para Entrenamiento (Data Augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        # Variación de color para que la red no se confunda con la luz (Fase Media)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # Rotación y flip para que la red aprenda la forma en cualquier ángulo
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transformaciones básicas para Validación (Sin alteración aleatoria)
    val_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Carga del dataset (espera carpetas: 0_Sano, 1_Fase_Inicial, 2_Fase_Media, 3_Fase_Final)
    try:
        # Se instancian dos dataset para aplicar diferentes transformaciones sin conflictos
        full_dataset_train = datasets.ImageFolder(root=data_dir, transform=train_transforms)
        full_dataset_val = datasets.ImageFolder(root=data_dir, transform=val_transforms)
    except FileNotFoundError:
        print(f"Error: No se encontró el dataset en {data_dir}. Asegúrese de crear las carpetas requeridas.")
        return None, None, []
        
    # Separación 80% Train, 20% Val con semilla fija para reproducibilidad
    dataset_size = len(full_dataset_train)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    # Generamos índices consistentes
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    
    # Asignamos los subconjuntos asegurando que Validación no tenga Data Augmentation
    train_dataset = Subset(full_dataset_train, indices[:train_size])
    val_dataset = Subset(full_dataset_val, indices[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = full_dataset_train.classes
    return train_loader, val_loader, class_names

# ==========================================
# BLOQUE 2: Arquitectura del Modelo
# ==========================================
def build_model(num_classes=4):
    """
    Carga SqueezeNet preentrenada y modifica el clasificador final.
    
    Explicación de Módulos Fire en SqueezeNet:
    - Capa Squeeze: Utiliza filtros 1x1 para comprimir la dimensión de los canales de entrada.
      Esto reduce drásticamente la cantidad de parámetros.
    - Capa Expand: Utiliza una combinación de filtros 1x1 y 3x3 para extraer características 
      en múltiples escalas espaciales a partir de los canales comprimidos.
    El resultado es un modelo altamente eficiente, logrando precisión similar a AlexNet 
    pero con menos de 5MB de tamaño, ideal para Edge AI.
    """
    # Cargar SqueezeNet versión 1.1 preentrenada con la sintaxis moderna (weights='DEFAULT')
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    
    # Modificar la capa de clasificación para adaptarla a nuestras clases (Fases de Sigatoka)
    # SqueezeNet classifier es una secuencia: [0: Dropout, 1: Conv2d, 2: ReLU, 3: AdaptiveAvgPool2d]
    # Modificamos la capa Conv2d (índice 1) para que produzca `num_classes` canales
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    
    # Ajustar num_classes en el modelo (propiedad interna de squeezenet)
    model.num_classes = num_classes
    
    return model

# ==========================================
# BLOQUE 3: Motor de Aprendizaje
# ==========================================
def setup_training(model):
    """
    Configura la función de pérdida y el optimizador.
    """
    # Función de pérdida para clasificación multiclase
    criterion = nn.CrossEntropyLoss()
    
    # Optimizador Adam con learning rate bajo (0.0001) para captar cambios 
    # sutiles de color (ej. diferencia entre Fase Media y Fase Final)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    return criterion, optimizer

# ==========================================
# BLOQUE 4: Bucle de Entrenamiento y Validación
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    """
    Bucle de entrenamiento que incluye el cálculo de error mediante Autograd.
    """
    # Nota: el modelo ya debe haber sido movido al device antes de inicializar el optimizador
    model.train() # Asegurar estado de entrenamiento inicial
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # --- Fase de Entrenamiento ---
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            # Backward pass y Autograd
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # --- Fase de Validación ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects / len(val_loader.dataset)
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}\n')

    return model

# ==========================================
# BLOQUE 5: Evaluación y Matriz de Confusión
# ==========================================
def evaluate_and_plot_confusion_matrix(model, val_loader, class_names, device='cpu'):
    """
    Genera predicciones finales y grafica la matriz de confusión usando sklearn.metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    
    # Graficar la matriz de confusión
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión - Detección de Sigatoka Negra')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Matriz de confusión guardada como 'confusion_matrix.png'.")
    print("Por favor, revise detenidamente las métricas entre 'Fase_Media' y 'Fase_Final'.")

# ==========================================
# BLOQUE 6: Ejecución Principal y Exportación
# ==========================================
def main():
    # Parámetros
    data_dir = 'dataset_sigatoka' # Cambia esto por la ruta real a tu dataset
    batch_size = 32
    num_epochs = 15
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo de cómputo: {device}")
    
    # 1. Setup Data
    train_loader, val_loader, class_names = create_dataloaders(data_dir, batch_size)
    if not train_loader:
        return
        
    print(f"Clases detectadas: {class_names}")
    
    # 2. Build Model y mover a dispositivo (CRÍTICO: hacer esto antes del optimizador)
    model = build_model(num_classes=len(class_names))
    model = model.to(device)
    
    # 3. Setup Engine
    criterion, optimizer = setup_training(model)
    
    # 4. Train Model
    print("Iniciando entrenamiento...")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    
    # 5. Evaluate
    print("Evaluando modelo y generando matriz de confusión...")
    evaluate_and_plot_confusion_matrix(trained_model, val_loader, class_names, device)
    
    # 6. Export Model
    save_path = 'sigatoka_squeezenet.pth'
    torch.save(trained_model.state_dict(), save_path)
    
    # Verificar el tamaño del archivo para Edge AI (< 5MB)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\nModelo guardado en: {save_path}")
    print(f"Tamaño del archivo: {file_size_mb:.2f} MB")
    
    if file_size_mb < 5.0:
        print("¡Éxito! El peso del archivo es eficiente para dispositivos Edge AI (< 5 MB).")
    else:
        print("Advertencia: El tamaño del archivo supera los 5 MB.")

if __name__ == '__main__':
    main()
