import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ==========================================
# BLOQUE 1: Carga del Modelo
# ==========================================
def cargar_modelo(ruta_pesos='sigatoka_squeezenet.pth'):
    """
    Construye la arquitectura SqueezeNet y carga los pesos entrenados.
    A diferencia de guardar todo el modelo, cargar el 'state_dict' (los pesos)
    es la práctica más recomendada y segura en PyTorch.
    """
    # 1. Instanciar la arquitectura base (SqueezeNet 1.1)
    model = models.squeezenet1_1(weights=None)
    
    # 2. Modificar el clasificador final para 4 clases (las fases de la enfermedad)
    model.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1,1))
    model.num_classes = 4
    
    # 3. Detectar si hay GPU disponible para acelerar el proceso
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 4. Cargar los pesos guardados en el archivo .pth
    if os.path.exists(ruta_pesos):
        # map_location asegura que se pueda cargar en CPU si se entrenó en GPU
        model.load_state_dict(torch.load(ruta_pesos, map_location=device))
    else:
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {ruta_pesos}")
    
    # 5. Mover el modelo al dispositivo (CPU o GPU) y ponerlo en modo evaluación
    model = model.to(device)
    model.eval() # IMPORTANTE: desactiva Dropout y otras capas de entrenamiento
    
    return model, device

# ==========================================
# BLOQUE 2: Preprocesamiento e Inferencia
# ==========================================
def predecir_sigatoka(ruta_imagen, model, device):
    """
    Carga una imagen nueva, le aplica las transformaciones necesarias
    y retorna el veredicto del modelo.
    """
    # Verificar si la imagen realmente existe en el disco
    if not os.path.exists(ruta_imagen):
        return f"Error: No se encontró la imagen en {ruta_imagen}"

    # Cargar la imagen y asegurar que tenga 3 canales (RGB)
    img = Image.open(ruta_imagen).convert('RGB') 
    
    # Aplicamos exactamente la misma transformación que usamos en Validación
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transformar la imagen y añadir la dimensión de batch: pasa de (C, H, W) a (1, C, H, W)
    img_t = transform(img).unsqueeze(0).to(device)

    # 3. La red "une los puntos" y da el veredicto
    # Usar torch.no_grad() ahorra memoria y tiempo, ya que no necesitamos calcular gradientes
    with torch.no_grad():
        outputs = model(img_t)
        # Obtener el índice de la clase con mayor probabilidad
        _, predicted = torch.max(outputs, 1)
    
    # Mapeo de resultados para que coincida con las carpetas de entrenamiento
    fases = ['0_Sano', '1_Fase_Inicial', '2_Fase_Media', '3_Fase_Final']
    return fases[int(predicted.item())]

# ==========================================
# BLOQUE 3: Ejecución de Ejemplo
# ==========================================
if __name__ == "__main__":
    try:
        print("Cargando el modelo...")
        modelo, dispositivo = cargar_modelo()
        print(f"Modelo cargado correctamente. Usando: {dispositivo}\n")
        
        # Aquí puedes cambiar el nombre por la imagen real que quieras evaluar
        imagen_prueba = 'prueba_campo.jpg' 
        
        # Si la imagen existe, hace la predicción, si no, advierte al usuario
        if os.path.exists(imagen_prueba):
            print(f"Analizando imagen: '{imagen_prueba}' ...")
            resultado = predecir_sigatoka(imagen_prueba, modelo, dispositivo)
            print(f"\nEl diagnóstico es: {resultado}")
        else:
            print(f"Por favor coloca una imagen llamada '{imagen_prueba}' en esta misma carpeta para probar el script.")
            
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
