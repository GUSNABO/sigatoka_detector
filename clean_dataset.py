import argparse
import hashlib
import os
import shutil

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SKIP_DIRS = {"_quarantine", "_balance_backup", ".git", "venv", ".venv"}


def iter_image_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.lower().endswith(IMAGE_EXTS):
                yield os.path.join(dirpath, name)


def is_image_valid(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def md5sum(path, chunk_size=1024 * 1024):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ahash(path, hash_size=8):
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    with Image.open(path) as img:
        img = img.convert("L").resize((hash_size, hash_size), resample)
        pixels = np.asarray(img, dtype=np.float32)
    avg = pixels.mean()
    bits = (pixels > avg).astype(np.uint8)
    return bits.tobytes()


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


def apply_action(path, action, root, quarantine_dir):
    if action == "delete":
        os.remove(path)
        return
    if action == "quarantine":
        rel_path = os.path.relpath(path, root)
        target = os.path.join(quarantine_dir, rel_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        target = unique_destination(target)
        shutil.move(path, target)


def main():
    parser = argparse.ArgumentParser(description="Limpia imagenes corruptas y duplicadas.")
    parser.add_argument("--root", default="dataset_sigatoka", help="Directorio del dataset")
    parser.add_argument(
        "--action",
        choices=["report", "delete", "quarantine"],
        default="report",
        help="Accion a realizar sobre archivos detectados",
    )
    parser.add_argument(
        "--quarantine-dir",
        default="dataset_sigatoka/_quarantine",
        help="Carpeta destino si action=quarantine",
    )
    parser.add_argument("--no-dedup", action="store_true", help="Desactiva deteccion de duplicados")
    parser.add_argument(
        "--dedup-method",
        choices=["md5", "ahash"],
        default="ahash",
        help="Metodo de deteccion de duplicados",
    )
    parser.add_argument("--ahash-size", type=int, default=8, help="Tamano del hash perceptual")
    args = parser.parse_args()

    root = args.root
    all_paths = list(iter_image_files(root))

    corrupted = []
    valid_paths = []
    for path in all_paths:
        if is_image_valid(path):
            valid_paths.append(path)
        else:
            corrupted.append(path)

    duplicates = []
    if not args.no_dedup:
        hash_to_path = {}
        for path in valid_paths:
            if args.dedup_method == "md5":
                hvalue = md5sum(path)
            else:
                hvalue = ahash(path, args.ahash_size)
            if hvalue in hash_to_path:
                duplicates.append(path)
            else:
                hash_to_path[hvalue] = path

    print("Resumen de limpieza")
    print(f"Total imagenes: {len(all_paths)}")
    print(f"Corruptas: {len(corrupted)}")
    print(f"Duplicadas: {len(duplicates)}")
    print(f"Accion: {args.action}")

    if args.action == "report":
        return

    targets = corrupted + duplicates
    for path in targets:
        apply_action(path, args.action, root, args.quarantine_dir)

    print(f"Archivos procesados: {len(targets)}")


if __name__ == "__main__":
    main()
