import argparse
import os
from icrawler.builtin import BingImageCrawler

DEFAULT_QUERIES = {
    "animales": ["gato", "perro", "vaca", "caballo"],
    "objetos": ["martillo", "silla", "telefono", "bicicleta"],
    "fondos": ["suelo", "cielo", "pared", "carretera"],
    "otras_plantas": ["hoja de mango", "hoja de cafe", "hoja de palma", "hoja de papaya"],
}


def distribute_counts(total, n):
    base = total // n
    remainder = total % n
    return [base + (1 if i < remainder else 0) for i in range(n)]


def download_category(category, queries, total_per_category, output_root):
    counts = distribute_counts(total_per_category, len(queries))
    for query, count in zip(queries, counts):
        safe_query = query.replace(" ", "_")
        target_dir = os.path.join(output_root, category, safe_query)
        os.makedirs(target_dir, exist_ok=True)
        existing = [
            name
            for name in os.listdir(target_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        file_idx_offset = len(existing)
        crawler = BingImageCrawler(storage={"root_dir": target_dir})
        crawler.crawl(keyword=query, max_num=count, file_idx_offset=file_idx_offset)


def main():
    parser = argparse.ArgumentParser(
        description="Descarga imagenes No_hoja en carpetas separadas usando icrawler."
    )
    parser.add_argument(
        "--output-root",
        default="dataset_sigatoka/4_No_hoja",
        help="Ruta base donde se guardaran las imagenes",
    )
    parser.add_argument(
        "--total-per-category",
        type=int,
        default=100,
        help="Numero total de imagenes por categoria",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=list(DEFAULT_QUERIES.keys()),
        help="Categorias a descargar: animales, objetos, fondos, otras_plantas",
    )
    args = parser.parse_args()

    for category in args.categories:
        if category not in DEFAULT_QUERIES:
            raise ValueError(f"Categoria no valida: {category}")

    for category in args.categories:
        queries = DEFAULT_QUERIES[category]
        download_category(category, queries, args.total_per_category, args.output_root)

    print("Descarga finalizada.")


if __name__ == "__main__":
    main()
