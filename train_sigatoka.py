import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==========================================
# BLOQUE 0: Utilidades para OOD (Fuera de Dominio)
# ==========================================
def l2_normalize(arr, axis=1, eps=1e-12):
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / (norm + eps)


def compute_class_centroids(features, labels, num_classes):
    centroids = []
    for cls in range(num_classes):
        cls_features = features[labels == cls]
        if cls_features.size == 0:
            centroids.append(np.zeros(features.shape[1], dtype=np.float32))
        else:
            centroids.append(cls_features.mean(axis=0))
    return np.vstack(centroids)


def compute_ood_threshold(val_features, val_labels, centroids, percentile=5):
    if val_features.size == 0:
        return 0.0
    feats_norm = l2_normalize(val_features, axis=1)
    cent_norm = l2_normalize(centroids, axis=1)
    true_centroids = cent_norm[val_labels]
    similarities = np.sum(feats_norm * true_centroids, axis=1)
    return float(np.percentile(similarities, percentile))

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
def evaluate_split(split_name, classifier, features, labels, class_names, output_prefix):
    preds = classifier.predict(features)

    recall_macro = recall_score(labels, preds, average="macro")
    f1_macro = f1_score(labels, preds, average="macro")
    recall_per_class = recall_score(labels, preds, average=None)

    print(f"\n[{split_name}] Recall Macro: {recall_macro:.4f}")
    print(f"[{split_name}] F1 Macro: {f1_macro:.4f}")
    print(f"[{split_name}] Recall por clase: {recall_per_class}")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title(f'Matriz de Confusión - {split_name}')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_{split_name}_cm.png')

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

    print("Extrayendo características con SqueezeNet...")
    extractor = build_feature_extractor(device)

    X_train, y_train = extract_features(train_loader, extractor, device)
    X_val, y_val = extract_features(val_loader, extractor, device)
    X_test, y_test = extract_features(test_loader, extractor, device)

    centroids = compute_class_centroids(X_train, y_train, len(class_names))
    ood_threshold = compute_ood_threshold(X_val, y_val, centroids, percentile=ood_percentile)
    print(f"Umbral OOD (percentil {ood_percentile}): {ood_threshold:.4f}")

    print("Entrenando clasificador SVM (prioridad en recall)...")
    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced"))
    ])
    classifier.fit(X_train, y_train)

    evaluate_split("Validacion", classifier, X_val, y_val, class_names, "sigatoka")
    evaluate_split("Test", classifier, X_test, y_test, class_names, "sigatoka")

    save_path = 'sigatoka_svm.joblib'
    joblib.dump(
        {
            "model": classifier,
            "class_names": class_names,
            "centroids": centroids,
            "ood_threshold": ood_threshold,
            "ood_percentile": ood_percentile,
        },
        save_path
    )
    print(f"\nModelo SVM guardado en: {save_path}")

if __name__ == '__main__':
    main()
