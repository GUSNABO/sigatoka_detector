import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ==========================================
# BLOQUE 1: Configuración de la Página
# ==========================================
# Esta configuración define el título en la pestaña del navegador y el icono
st.set_page_config(page_title="Detector de Sigatoka Negra", page_icon="🍌", layout="centered")

st.title("🍌 IA para Detección de Sigatoka Negra")
st.write("Sube una imagen de una hoja de banano para que el **Ensamblador** analice su textura y forma.")

# ==========================================
# BLOQUE 2: Carga del Modelo (con Caché)
# ==========================================
# El decorador @st.cache_resource asegura que el modelo se cargue en la RAM
# solo la primera vez que se abre la app, haciendo la experiencia mucho más rápida
@st.cache_resource 
def cargar_modelo():
    """
    Construye la arquitectura SqueezeNet y carga los pesos desde el archivo .pth.
    """
    # 1. Definir la arquitectura para que coincida con tu entrenamiento original
    model = models.squeezenet1_1(weights=None)
    
    # Modificar la capa final para predecir las 4 clases de Sigatoka
    model.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1,1))
    
    # 2. Cargar los pesos de tu archivo (los 2.78 MB). Se fuerza en CPU 
    # para asegurar que funcione en computadoras sin tarjeta de video
    try:
        model.load_state_dict(torch.load('sigatoka_squeezenet.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("Error: No se encontró 'sigatoka_squeezenet.pth'. Verifica que el archivo de pesos esté en la misma carpeta.")
        return None
        
    # 3. Poner en modo evaluación para apagar comportamientos de entrenamiento
    model.eval()
    return model

# ==========================================
# BLOQUE 3: Preprocesamiento de la Imagen
# ==========================================
def preprocesar_imagen(imagen):
    """
    Aplica las transformaciones para que la imagen nueva sea compatible
    con lo que espera la red (resolución 227x227 y normalización).
    """
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Convertir a RGB previene errores si el usuario sube un PNG con canal transparente (RGBA)
    img_rgb = imagen.convert('RGB')
    
    # unsqueeze(0) añade la dimensión del lote (batch), vital para que PyTorch lo procese
    return transform(img_rgb).unsqueeze(0)

# ==========================================
# BLOQUE 4: Interfaz de Usuario Principal
# ==========================================
# Instanciar el modelo usando la función con caché
model = cargar_modelo()

if model is not None:
    # Widget de Streamlit que permite al usuario cargar imágenes
    archivo_subido = st.file_uploader("Elige una imagen de la hoja...", type=["jpg", "jpeg", "png"])

    if archivo_subido is not None:
        # Abrir y mostrar la imagen subida
        imagen = Image.open(archivo_subido)
        # use_container_width adapta la imagen al ancho de la interfaz (use_column_width está deprecado)
        st.image(imagen, caption='Imagen subida para análisis', use_container_width=True)
        
        # Muestra un indicador visual de carga mientras procesa
        with st.spinner("Analizando características visuales con el Ensamblador de SqueezeNet..."):
            # 1. Preparar la imagen (el mismo proceso de la tesis)
            img_t = preprocesar_imagen(imagen)
            
            # 2. Ejecutar la predicción
            with torch.no_grad():
                output = model(img_t)
                _, prediccion = torch.max(output, 1)

            # 3. Mostrar resultado basado en las carpetas originales
            fases = ['0_Sano', '1_Fase_Inicial', '2_Fase_Media', '3_Fase_Final']
            resultado = fases[int(prediccion.item())]
        
        # Muestra un recuadro verde con el diagnóstico final
        st.success(f"Diagnóstico de la IA: **{resultado}**")
        
        # ==========================================
        # BLOQUE 5: Explicación Científica (Sustentación)
        # ==========================================
        st.markdown("### Explicación del Ensamblador")
        if "Sano" in resultado:
            st.info("El modelo detecta una superficie foliar homogénea, sin presencia de estrías cloróticas ni evidencia de necrosis.")
        elif "Inicial" in resultado:
            st.warning("El **Ensamblador** identificó posibles estrías o manchas pálidas sutiles, indicativas de la Fase Inicial de la infección.")
        elif "Media" in resultado:
            st.error("El **Ensamblador** detectó textura seca (1x1) y bordes irregulares (3x3) típicos de la necrosis marrón característica de la Fase Media.")
        elif "Final" in resultado:
            st.error("El **Ensamblador** ha identificado un colapso foliar masivo y manchas necróticas oscuras fusionadas, propias del estado terminal o Fase Final.")
