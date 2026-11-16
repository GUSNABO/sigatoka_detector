import argparse
import os
import random
import shutil

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SKIP_DIRS = {"_quarantine", "_balance_backup", ".git", "venv", ".venv"}


def iter_image_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.lower().endswith(IMAGE_EXTS):
                yield os.path.join(dirpath, name)


def unique_destination(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    idx = 1
    while True:
        candidate = f"{base}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def main():
    parser = argparse.ArgumentParser(description="Balancea clases del dataset por cantidad.")
    parser.add_argument("--root", default="dataset_sigatoka", help="Directorio del dataset")
    parser.add_argument("--target", type=int, default=None, help="Cantidad objetivo por clase")
    parser.add_argument(
        "--action",
        choices=["report", "apply"],
        default="report",
        help="Solo reportar o aplicar balanceo",
    )
    parser.add_argument(
        "--backup-dir",
        default="dataset_sigatoka/_balance_backup",
        help="Carpeta destino para imagenes removidas",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para muestreo aleatorio")
    parser.add_argument("--classes", nargs="*", default=None, help="Clases a balancear")
    args = parser.parse_args()

    root = args.root
    class_names = [
        name
        for name in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, name)) and not name.startswith("_")
    ]
    if args.classes:
        class_names = [name for name in class_names if name in args.classes]

    class_files = {}
    for class_name in class_names:
        class_path = os.path.join(root, class_name)
        class_files[class_name] = list(iter_image_files(class_path))

    if not class_files:
        print("No se encontraron clases para balancear.")
        return

    counts = {name: len(files) for name, files in class_files.items()}
    target = args.target if args.target is not None else min(counts.values())

    print("Resumen de balanceo")
    print(f"Objetivo por clase: {target}")
    for name in class_names:
        print(f"- {name}: {counts[name]}")

    rng = random.Random(args.seed)
    to_move = []
    for name, files in class_files.items():
        if len(files) <= target:
            continue
        extras = rng.sample(files, len(files) - target)
        to_move.extend(extras)

    print(f"Archivos a mover: {len(to_move)}")

    if args.action == "report":
        return

    for path in to_move:
        rel_path = os.path.relpath(path, root)
        target_path = os.path.join(args.backup_dir, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        target_path = unique_destination(target_path)
        shutil.move(path, target_path)

    print("Balanceo aplicado.")


if __name__ == "__main__":
    main()
