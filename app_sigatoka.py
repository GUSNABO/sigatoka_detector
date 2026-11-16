import os
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import joblib
import numpy as np

# ==========================================
# BLOQUE 1: Configuración de la Página
# ==========================================
# Esta configuración define el título en la pestaña del navegador y el icono
st.set_page_config(page_title="Detector de Sigatoka Negra", page_icon="🍌", layout="centered")

st.title("Detección de Sigatoka Negra")
st.write("Sube una imagen de una hoja de banano para que el **Ensamblador** analice su textura y forma.")

if st.button("Recargar modelo"):
    st.cache_resource.clear()
    st.rerun()

# ==========================================
# BLOQUE 2: Carga del Modelo Hibrido (con Caché)
# ==========================================
# El decorador @st.cache_resource asegura que el modelo se cargue en la RAM
# solo la primera vez que se abre la app, haciendo la experiencia mucho más rápida
@st.cache_resource 
def cargar_modelo_hibrido():
    """
    Carga el clasificador SVM y el extractor SqueezeNet para inferencia.
    """
    if not os.path.exists('sigatoka_svm.joblib'):
        st.error("Error: No se encontró 'sigatoka_svm.joblib'. Ejecuta train_sigatoka.py primero.")
        return None

    artifact = joblib.load('sigatoka_svm.joblib')
    classifier = artifact.get("model")
    class_names = artifact.get("class_names")
    centroids = artifact.get("centroids")
    ood_threshold = artifact.get("ood_threshold")

    if classifier is None or class_names is None:
        st.error("Error: El archivo joblib no contiene 'model' o 'class_names'.")
        return None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extractor = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    extractor.eval()
    extractor = extractor.to(device)

    return classifier, class_names, extractor, device, centroids, ood_threshold

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


def extraer_features(img_t, extractor, device):
    img_t = img_t.to(device)
    with torch.no_grad():
        outputs = extractor.features(img_t)
        pooled = F.adaptive_avg_pool2d(outputs, (1, 1))
        features = pooled.view(pooled.size(0), -1).cpu().numpy()
    return features


def detectar_ood(features, centroids, threshold):
    if centroids is None or threshold is None:
        return False, None

    features = np.asarray(features, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    cent_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    similarities = np.dot(feat_norm, cent_norm.T)
    max_sim = float(np.max(similarities))

    return max_sim < threshold, max_sim


def es_no_hoja_label(label):
    if not isinstance(label, str):
        return False
    normalized = label.lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return "no hoja" in normalized

# ==========================================
# BLOQUE 4: Interfaz de Usuario Principal
# ==========================================
# Instanciar el modelo usando la función con caché
model_bundle = cargar_modelo_hibrido()

if model_bundle is not None:
    classifier, class_names, extractor, device, centroids, ood_threshold = model_bundle
    if centroids is None or ood_threshold is None:
        st.warning("El modelo no incluye el filtro OOD. Reentrena con la ultima version de train_sigatoka.py.")
    if not any(es_no_hoja_label(name) for name in class_names):
        st.warning("El modelo no tiene la clase No_hoja. Reentrena incluyendo esa carpeta.")
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
            
            # 2. Evaluar si la imagen es OOD (fuera del dominio de hojas de banano)
            features = extraer_features(img_t, extractor, device)
            es_ood, _max_sim = detectar_ood(features, centroids, ood_threshold)

            if es_ood:
                st.error("La imagen parece estar fuera del dominio de hojas de banano. No se emite un diagnóstico.")
                st.info("Prueba con una foto centrada en una hoja de banano, con buena iluminación y sin fondo dominante.")
            else:
                # 3. Ejecutar la predicción
                prediccion = classifier.predict(features)[0]

                # 4. Mostrar resultado basado en las carpetas originales
                resultado = class_names[int(prediccion)]

                if es_no_hoja_label(resultado):
                    st.warning("La imagen no corresponde a una hoja de banano. No se emite un diagnóstico.")
                else:
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
