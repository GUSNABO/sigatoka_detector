import argparse
import os
from collections import Counter

from PIL import Image

from predict_sigatoka import (
    cargar_modelo,
    detectar_ood,
    es_no_hoja_label,
    extraer_features,
    preprocesar_imagen,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def iter_image_files(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(IMAGE_EXTS):
                yield os.path.join(dirpath, name)


def main():
    parser = argparse.ArgumentParser(description="Evalua falsos positivos en No_hoja.")
    parser.add_argument("--neg-dir", default="dataset_sigatoka/4_No_hoja", help="Carpeta negativa")
    parser.add_argument("--model", default="sigatoka_svm.joblib", help="Modelo entrenado")
    parser.add_argument("--limit", type=int, default=None, help="Limitar numero de imagenes")
    args = parser.parse_args()

    classifier, class_names, extractor, device, centroids, ood_threshold, class_sim_thresholds = cargar_modelo(args.model)
    has_no_hoja = any(es_no_hoja_label(name) for name in class_names)

    total_files = 0
    processed = 0
    failed = 0
    counts = Counter()
    breakdown = Counter()

    for path in iter_image_files(args.neg_dir):
        total_files += 1
        if args.limit and processed >= args.limit:
            break
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            failed += 1
            continue

        processed += 1
        img_t = preprocesar_imagen(img)
        features = extraer_features(img_t, extractor, device)

        es_ood, _, _ = detectar_ood(features, centroids, ood_threshold)
        if es_ood:
            counts["ood"] += 1
            continue

        pred = classifier.predict(features)[0]
        label = class_names[int(pred)]

        if es_no_hoja_label(label):
            counts["no_hoja"] += 1
        else:
            counts["sigatoka"] += 1
            breakdown[label] += 1

    print("Resumen de evaluacion No_hoja")
    print(f"Total archivos: {total_files}")
    print(f"Procesados: {processed}")
    print(f"Fallidos: {failed}")
    print(f"OOD rechazados: {counts['ood']}")
    print(f"Predicho No_hoja: {counts['no_hoja']}")
    print(f"Falsos positivos Sigatoka: {counts['sigatoka']}")

    if processed > 0:
        fp_rate = counts["sigatoka"] / processed
        ood_rate = counts["ood"] / processed
        no_hoja_rate = counts["no_hoja"] / processed
        print(f"Tasa falsos positivos: {fp_rate:.4f}")
        print(f"Tasa OOD: {ood_rate:.4f}")
        print(f"Tasa No_hoja: {no_hoja_rate:.4f}")

    if not has_no_hoja:
        print("Advertencia: el modelo no tiene la clase No_hoja.")

    if breakdown:
        print("\nFalsos positivos por clase:")
        for label, count in breakdown.most_common():
            print(f"- {label}: {count}")


if __name__ == "__main__":
    main()
