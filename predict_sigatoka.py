import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import joblib
import numpy as np

# ==========================================
# BLOQUE 1: Carga del Modelo Hibrido
# ==========================================
def cargar_modelo(ruta_modelo='sigatoka_svm.joblib'):
    """
    Carga el clasificador SVM y construye el extractor SqueezeNet.
    """
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {ruta_modelo}")

    artifact = joblib.load(ruta_modelo)
    classifier = artifact.get("model")
    class_names = artifact.get("class_names")
    centroids = artifact.get("centroids")
    ood_threshold = artifact.get("ood_threshold")

    if classifier is None or class_names is None:
        raise ValueError("El archivo joblib no contiene 'model' o 'class_names'.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extractor = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    extractor.eval()
    extractor = extractor.to(device)

    return classifier, class_names, extractor, device, centroids, ood_threshold

# ==========================================
# BLOQUE 2: Preprocesamiento y Extraccion
# ==========================================
def preprocesar_imagen(imagen):
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(imagen.convert('RGB')).unsqueeze(0)


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


def predecir_sigatoka(ruta_imagen, classifier, class_names, extractor, device, centroids, ood_threshold):
    if not os.path.exists(ruta_imagen):
        return f"Error: No se encontró la imagen en {ruta_imagen}"

    img = Image.open(ruta_imagen).convert('RGB')
    img_t = preprocesar_imagen(img)
    features = extraer_features(img_t, extractor, device)

    es_ood, _max_sim = detectar_ood(features, centroids, ood_threshold)
    if es_ood:
        return "Fuera de dominio: la imagen no parece una hoja de banano."

    pred = classifier.predict(features)[0]
    resultado = class_names[int(pred)]
    if es_no_hoja_label(resultado):
        return "Sin diagnostico: la imagen no corresponde a una hoja de banano."

    return resultado

# ==========================================
# BLOQUE 3: Ejecución de Ejemplo
# ==========================================
if __name__ == "__main__":
    try:
        print("Cargando el modelo hibrido (SqueezeNet + SVM)...")
        classifier, class_names, extractor, device, centroids, ood_threshold = cargar_modelo()
        print(f"Modelo cargado correctamente. Usando: {device}\n")

        imagen_prueba = 'prueba_campo.jpg'

        if os.path.exists(imagen_prueba):
            print(f"Analizando imagen: '{imagen_prueba}' ...")
            resultado = predecir_sigatoka(
                imagen_prueba,
                classifier,
                class_names,
                extractor,
                device,
                centroids,
                ood_threshold
            )
            print(f"\nEl diagnóstico es: {resultado}")
        else:
            print(f"Por favor coloca una imagen llamada '{imagen_prueba}' en esta misma carpeta para probar el script.")

    except Exception as e:
        print(f"Error durante la ejecución: {e}")
