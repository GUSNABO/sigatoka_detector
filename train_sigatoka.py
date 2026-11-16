import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==========================================
# BLOQUE 0: Utilidades para OOD (Fuera de Dominio)
# ==========================================
OOD_CLASS_IDX = 4  # Clase 4: No Hoja (Fuera del dominio de hojas de banano)


def l2_normalize(arr, axis=1, eps=1e-12):
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / (norm + eps)


def compute_class_centroids(features, labels, num_classes, exclude_class=OOD_CLASS_IDX):
    """
    Calcula centroides normalizados L2 para las clases pertenecientes al dominio (hojas).
    """
    feats_norm = l2_normalize(features, axis=1)
    centroids = []
    for cls in range(num_classes):
        if cls == exclude_class:
            continue
        cls_features = feats_norm[labels == cls]
        if cls_features.size == 0:
            centroids.append(np.zeros(features.shape[1], dtype=np.float32))
        else:
            centroids.append(cls_features.mean(axis=0))
    return l2_normalize(np.vstack(centroids), axis=1)


def compute_ood_threshold(val_features, val_labels, centroids, percentile=5, exclude_class=OOD_CLASS_IDX):
    """
    Calcula el umbral de similitud coseno usando únicamente muestras válidas del dominio.
    """
    if val_features.size == 0:
        return 0.0
    in_domain_mask = val_labels != exclude_class
    val_feats_in = val_features[in_domain_mask]
    val_labels_in = val_labels[in_domain_mask]

    if val_feats_in.size == 0:
        return 0.0

    feats_norm = l2_normalize(val_feats_in, axis=1)
    cent_norm = l2_normalize(centroids, axis=1)
    true_centroids = cent_norm[val_labels_in]
    similarities = np.sum(feats_norm * true_centroids, axis=1)
    return float(np.percentile(similarities, percentile))


def predict_with_ood(classifier, features, centroids, ood_threshold, ood_class_idx=OOD_CLASS_IDX):
    """
    Compuerta OOD:
    1. Calcula la máxima similitud coseno de cada vector 512D contra los centroides de hoja.
    2. Si max_sim < ood_threshold -> Rechaza la imagen como Clase 4 (No Hoja) sin pasar por la SVM.
    3. Solo las muestras dentro del dominio son clasificadas por la SVM.
    """
    feats_norm = l2_normalize(features, axis=1)
    cent_norm = l2_normalize(centroids, axis=1)

    sim_matrix = np.dot(feats_norm, cent_norm.T)
    max_sim = np.max(sim_matrix, axis=1)

    preds = np.full(features.shape[0], ood_class_idx, dtype=np.int64)
    in_domain_mask = max_sim >= ood_threshold
    if np.any(in_domain_mask):
        preds[in_domain_mask] = classifier.predict(features[in_domain_mask])

    return preds, max_sim, in_domain_mask

# ==========================================
# BLOQUE 1: Configuración y Preprocesamiento
# ==========================================
def create_datasets(data_dir, val_size=0.15, test_size=0.15, seed=42):
    """
    Crea splits estratificados (train/val/test) y aplica transformaciones diferenciadas.
    """
    try:
        base_dataset = datasets.ImageFolder(root=data_dir)
    except FileNotFoundError:
        print(f"Error: No se encontró el dataset en {data_dir}. Asegúrese de crear las carpetas requeridas.")
        return None, None, None, []

    class_names = base_dataset.classes
    targets = np.array(base_dataset.targets)
    indices = np.arange(len(targets))

    temp_size = val_size + test_size
    train_idx, temp_idx, _, temp_y = train_test_split(
        indices,
        targets,
        test_size=temp_size,
        stratify=targets,
        random_state=seed
    )

    test_ratio = test_size / temp_size
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio,
        stratify=temp_y,
        random_state=seed
    )

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(227, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_train = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    dataset_eval = datasets.ImageFolder(root=data_dir, transform=eval_transforms)

    train_dataset = Subset(dataset_train, train_idx)
    val_dataset = Subset(dataset_eval, val_idx)
    test_dataset = Subset(dataset_eval, test_idx)

    return train_dataset, val_dataset, test_dataset, class_names


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

# ==========================================
# BLOQUE 2: Extractor de Características
# ==========================================
def build_feature_extractor(device):
    """
    Carga SqueezeNet preentrenada y deja solo el extractor de características.
    """
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    model.eval()
    model = model.to(device)
    return model


def extract_features(dataloader, model, device):
    """
    Extrae embeddings de 512 dimensiones usando SqueezeNet + Global Average Pooling.
    """
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model.features(inputs)
            pooled = F.adaptive_avg_pool2d(outputs, (1, 1))
            batch_features = pooled.view(pooled.size(0), -1).cpu().numpy()

            features.append(batch_features)
            labels.append(targets.numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

# ==========================================
# BLOQUE 3: Entrenamiento SVM y Evaluacion
# ==========================================
def evaluate_split(split_name, classifier, features, labels, class_names, output_prefix, centroids=None, ood_threshold=None):
    if centroids is not None and ood_threshold is not None and len(class_names) > OOD_CLASS_IDX:
        preds, _, in_domain_mask = predict_with_ood(
            classifier, features, centroids, ood_threshold, ood_class_idx=OOD_CLASS_IDX
        )
        rejected_count = int(np.sum(~in_domain_mask))
        print(f"\n[{split_name}] Imágenes rechazadas por filtro OOD (Clase 4 - No Hoja): {rejected_count}/{len(features)}")
    else:
        preds = classifier.predict(features)

    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    print(f"\n--- MÉTRICAS DETALLADAS ({split_name}) - ENFOQUE DETECCIÓN TEMPRANA ---")
    print(f"[{split_name}] Precision Macro: {precision_macro:.4f}")
    print(f"[{split_name}] Recall Macro:    {recall_macro:.4f}")
    print(f"[{split_name}] F1-Score Macro:  {f1_macro:.4f}")

    print("\nDesglose por clase (Precisión | Recall | F1-Score):")
    for idx, cls_name in enumerate(class_names):
        if idx < len(precision_per_class):
            print(
                f"  - Clase {idx} ({cls_name:<18}): "
                f"P={precision_per_class[idx]:.4f} | "
                f"R={recall_per_class[idx]:.4f} | "
                f"F1={f1_per_class[idx]:.4f}"
            )

    print("\nReporte de Clasificación Completo:")
    print(classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title(f'Matriz de Confusión - {split_name}')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_{split_name}_cm.png')
    plt.close()


def benchmark_edge_performance(extractor, classifier, centroids, ood_threshold, test_dataset, device, svm_save_path, num_samples=100):
    """
    Mide tiempos de inferencia por imagen (ms) y peso de los modelos exportados para Edge AI.
    """
    extractor.eval()
    n_eval = min(num_samples, len(test_dataset))
    single_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dummy_input = torch.randn(1, 3, 227, 227, device=device)
    with torch.no_grad():
        _ = extractor.features(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_cnn_ms = []
    times_ood_svm_ms = []
    times_total_ms = []

    with torch.no_grad():
        for i, (img_tensor, _) in enumerate(single_loader):
            if i >= n_eval:
                break

            t_start = time.perf_counter()

            # 1. Extracción de características 512D con SqueezeNet
            img_tensor = img_tensor.to(device)
            outputs = extractor.features(img_tensor)
            pooled = F.adaptive_avg_pool2d(outputs, (1, 1))
            feat_512 = pooled.view(1, -1).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_cnn = time.perf_counter()

            # 2. Filtro OOD + Clasificación SVM
            _ = predict_with_ood(classifier, feat_512, centroids, ood_threshold)
            t_end = time.perf_counter()

            times_cnn_ms.append((t_cnn - t_start) * 1000.0)
            times_ood_svm_ms.append((t_end - t_cnn) * 1000.0)
            times_total_ms.append((t_end - t_start) * 1000.0)

    svm_size_kb = os.path.getsize(svm_save_path) / 1024.0
    svm_size_mb = svm_size_kb / 1024.0

    squeezenet_params_bytes = sum(
        p.numel() * p.element_size() for p in extractor.features.parameters()
    )
    squeezenet_size_mb = squeezenet_params_bytes / (1024.0 * 1024.0)
    total_pipeline_mb = squeezenet_size_mb + svm_size_mb

    mean_total_ms = np.mean(times_total_ms)
    fps = 1000.0 / mean_total_ms if mean_total_ms > 0 else 0.0

    print("\n" + "=" * 60)
    print("REPORTE DE RENDIMIENTO EDGE AI (DESPLIEGUE LOCAL)")
    print("=" * 60)
    print(f"Muestras evaluadas (batch_size=1): {n_eval} imágenes")
    print(f"Tiempo medio Extracción SqueezeNet (512D): {np.mean(times_cnn_ms):.2f} ± {np.std(times_cnn_ms):.2f} ms/img")
    print(f"Tiempo medio Filtro OOD + SVM:             {np.mean(times_ood_svm_ms):.2f} ± {np.std(times_ood_svm_ms):.2f} ms/img")
    print(f"Tiempo TOTAL de inferencia por imagen:     {mean_total_ms:.2f} ± {np.std(times_total_ms):.2f} ms/img")
    print(f"Throughput estimado:                       {fps:.1f} imágenes/segundo (FPS)")
    print("-" * 60)
    print(f"Peso Extractor SqueezeNet (features):      {squeezenet_size_mb:.2f} MB")
    print(f"Peso Clasificador SVM + Centroides OOD:    {svm_size_kb:.2f} KB ({svm_size_mb:.4f} MB)")
    print(f"Peso TOTAL del Pipeline Híbrido:           {total_pipeline_mb:.2f} MB")
    print("=" * 60 + "\n")

# ==========================================
# BLOQUE 4: Ejecución Principal y Exportación
# ==========================================
def main():
    data_dir = 'dataset_sigatoka'
    batch_size = 32
    val_size = 0.15
    test_size = 0.15
    ood_percentile = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Usando dispositivo de cómputo: {device}")

    train_dataset, val_dataset, test_dataset, class_names = create_datasets(
        data_dir,
        val_size=val_size,
        test_size=test_size
    )
    if not train_dataset:
        return

    print(f"Clases detectadas: {class_names}")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size
    )

    print("\n" + "="*50)
    print(f"Total de imágenes en el Train Loader: {len(train_loader.dataset)}")
    print("="*50 + "\n")

    print("Extrayendo características con SqueezeNet...")
    extractor = build_feature_extractor(device)

    X_train, y_train = extract_features(train_loader, extractor, device)
    X_val, y_val = extract_features(val_loader, extractor, device)
    X_test, y_test = extract_features(test_loader, extractor, device)

    # === CÓDIGO NUEVO A AGREGAR PARA LA EVIDENCIA DE LA TESIS (OE2) ===
    print("\n--- COMPROBACIÓN DE EXTRACCIÓN SQUEEZENET (512 CARACTERÍSTICAS) ---")
    print(f"Dimensión original por imagen: 227 x 227 x 3 = {227*227*3} valores")
    print(f"Matriz X_train extraída: {X_train.shape} (Muestras, Características)")
    print(f"Matriz X_val extraída:   {X_val.shape} (Muestras, Características)")
    print(f"Matriz X_test extraída:  {X_test.shape} (Muestras, Características)")
    print(f"Verificación de valores nulos (NaN) en X_train: {np.isnan(X_train).sum()}")
    print(f"Muestra del vector (primeros 5 valores) de la imagen 0:\n{X_train[0, :5]}\n")
    # ====================================================================

    centroids = compute_class_centroids(X_train, y_train, len(class_names))
    ood_threshold = compute_ood_threshold(X_val, y_val, centroids, percentile=ood_percentile)
    print(f"Umbral OOD (percentil {ood_percentile}): {ood_threshold:.4f}")

    print("Entrenando clasificador SVM (prioridad en recall)...")
    in_domain_train = y_train != OOD_CLASS_IDX
    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced"))
    ])
    classifier.fit(X_train[in_domain_train], y_train[in_domain_train])

    evaluate_split("Validacion", classifier, X_val, y_val, class_names, "sigatoka", centroids, ood_threshold)
    evaluate_split("Test", classifier, X_test, y_test, class_names, "sigatoka", centroids, ood_threshold)

    save_path = 'sigatoka_svm.joblib'
    joblib.dump(
        {
            "model": classifier,
            "class_names": class_names,
            "centroids": centroids,
            "ood_threshold": ood_threshold,
            "ood_percentile": ood_percentile,
            "ood_class_idx": OOD_CLASS_IDX,
        },
        save_path
    )
    print(f"\nModelo SVM guardado en: {save_path}")

    benchmark_edge_performance(
        extractor=extractor,
        classifier=classifier,
        centroids=centroids,
        ood_threshold=ood_threshold,
        test_dataset=test_dataset,
        device=device,
        svm_save_path=save_path,
    )

if __name__ == '__main__':
    main()
