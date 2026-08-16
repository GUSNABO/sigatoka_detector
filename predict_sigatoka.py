import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import joblib
import numpy as np

NO_HOJA_MARGIN = 0.02

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
    class_sim_thresholds = artifact.get("class_sim_thresholds") or {}

    if classifier is None or class_names is None:
        raise ValueError("El archivo joblib no contiene 'model' o 'class_names'.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extractor = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    extractor.eval()
    extractor = extractor.to(device)

    return classifier, class_names, extractor, device, centroids, ood_threshold, class_sim_thresholds

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


def calcular_similitudes(features, centroids):
    if centroids is None:
        return None

    features = np.asarray(features, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    cent_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    similarities = np.dot(feat_norm, cent_norm.T)
    return similarities[0]


def detectar_ood(features, centroids, threshold):
    if centroids is None or threshold is None:
        return False, None, None

    similarities = calcular_similitudes(features, centroids)
    if similarities is None:
        return False, None, None

    max_sim = float(np.max(similarities))
    return max_sim < threshold, max_sim, similarities


def es_no_hoja_label(label):
    if not isinstance(label, str):
        return False
    normalized = label.lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return "no hoja" in normalized


def mejor_sigatoka(similarities, class_names):
    if similarities is None:
        return None, None

    indices = [
        idx for idx, name in enumerate(class_names)
        if not es_no_hoja_label(name)
    ]
    if not indices:
        return None, None

    sig_sims = similarities[indices]
    best_pos = int(np.argmax(sig_sims))
    best_idx = indices[best_pos]
    return best_idx, float(sig_sims[best_pos])


def predecir_sigatoka(
    ruta_imagen,
    classifier,
    class_names,
    extractor,
    device,
    centroids,
    ood_threshold,
    class_sim_thresholds,
):
    if not os.path.exists(ruta_imagen):
        return f"Error: No se encontró la imagen en {ruta_imagen}"

    img = Image.open(ruta_imagen).convert('RGB')
    img_t = preprocesar_imagen(img)
    features = extraer_features(img_t, extractor, device)

    pred = classifier.predict(features)[0]
    resultado = class_names[int(pred)]

    es_ood, _max_sim, similarities = detectar_ood(features, centroids, ood_threshold)

    if es_no_hoja_label(resultado):
        no_hoja_threshold = class_sim_thresholds.get(resultado)
        no_hoja_idx = class_names.index(resultado) if resultado in class_names else None
        no_hoja_sim = similarities[no_hoja_idx] if (similarities is not None and no_hoja_idx is not None) else None
        best_sig_idx, best_sig_sim = mejor_sigatoka(similarities, class_names)
        sig_thresh = None

        if best_sig_idx is not None:
            sig_thresh = class_sim_thresholds.get(class_names[best_sig_idx], ood_threshold)
        if sig_thresh is None:
            sig_thresh = 0.0

        threshold_override = (
            no_hoja_threshold is not None
            and no_hoja_sim is not None
            and no_hoja_sim < no_hoja_threshold
        )
        margin_override = (
            no_hoja_sim is not None
            and best_sig_sim is not None
            and (best_sig_sim + NO_HOJA_MARGIN >= no_hoja_sim)
        )

        if (
            best_sig_idx is not None
            and best_sig_sim is not None
            and best_sig_sim >= sig_thresh
            and (threshold_override or margin_override)
        ):
            resultado = class_names[best_sig_idx]
            return f"Diagnostico con baja confianza: {resultado}"

        return "Sin diagnostico: la imagen no corresponde a una hoja de banano."

    if es_ood:
        return f"Diagnostico con baja confianza: {resultado}"

    return resultado

# ==========================================
# BLOQUE 3: Ejecución de Ejemplo
# ==========================================
if __name__ == "__main__":
    try:
        print("Cargando el modelo hibrido (SqueezeNet + SVM)...")
        classifier, class_names, extractor, device, centroids, ood_threshold, class_sim_thresholds = cargar_modelo()
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
                ood_threshold,
                class_sim_thresholds
            )
            print(f"\nEl diagnóstico es: {resultado}")
        else:
            print(f"Por favor coloca una imagen llamada '{imagen_prueba}' en esta misma carpeta para probar el script.")

    except Exception as e:
        print(f"Error durante la ejecución: {e}")
