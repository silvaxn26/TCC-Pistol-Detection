from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from dataset_roots import (
    COCO_BASE_DIR,
    SPLITS,
    YOLO_BASE_DIR,
    find_coco_annotation,
    find_coco_dataset_root,
    find_yolo_dataset_root,
    format_detected_root,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT_DIR / "results" / "reports"
AMBIGUOUS_CLASSES = {"0", "1"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yolo_class_names(yolo_dir: Path) -> list[str]:
    data_yaml_path = yolo_dir / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {data_yaml_path}")
    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=lambda value: int(value))]
    return list(names)


def read_coco_categories(coco_dir: Path | None) -> dict[str, list[str]]:
    categories_by_split = {}
    if coco_dir is None:
        return {split: [] for split in SPLITS}
    for split in SPLITS:
        annotation_path = find_coco_annotation(coco_dir / split)
        if not annotation_path:
            categories_by_split[split] = []
            continue
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories_by_split[split] = sorted(category["name"] for category in data.get("categories", []))
    return categories_by_split


def find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    yolo_dir = find_yolo_dataset_root()
    coco_dir = find_coco_dataset_root()
    print(format_detected_root("YOLOv8", YOLO_BASE_DIR, yolo_dir))
    print(format_detected_root("COCO", COCO_BASE_DIR, coco_dir))
    if yolo_dir is None:
        print("Dataset YOLOv8 nao encontrado. Execute primeiro:")
        print("py src/dataset/download_roboflow_yolo.py")
        return

    class_names = read_yolo_class_names(yolo_dir)
    coco_categories_by_split = read_coco_categories(coco_dir)
    annotation_counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    ambiguous_examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    print(f"Classes encontradas no YOLO data.yaml: {class_names}")
    print("\nCategorias encontradas nos arquivos COCO:")
    for split, categories in coco_categories_by_split.items():
        print(f"- {split}: {categories if categories else 'nao encontrado'}")

    yolo_class_set = set(class_names)
    coco_class_set = set().union(*(set(categories) for categories in coco_categories_by_split.values()))
    only_yolo = sorted(yolo_class_set - coco_class_set)
    only_coco = sorted(coco_class_set - yolo_class_set)
    print("\nComparacao YOLOv8 x COCO:")
    print(f"- Classes em ambos: {sorted(yolo_class_set & coco_class_set)}")
    print(f"- Somente no YOLOv8: {only_yolo}")
    print(f"- Somente no COCO: {only_coco}")

    for split in SPLITS:
        labels_dir = yolo_dir / split / "labels"
        images_dir = yolo_dir / split / "images"
        if not labels_dir.exists():
            print(f"Aviso: labels nao encontrados para split {split}: {labels_dir}")
            continue

        for label_path in labels_dir.glob("*.txt"):
            classes_in_image = set()
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"INVALID_{class_id}"
                annotation_counts[class_name] += 1
                classes_in_image.add(class_name)
                if class_name in AMBIGUOUS_CLASSES and len(ambiguous_examples[class_name]) < 10:
                    image_path = find_image_for_label(images_dir, label_path.stem)
                    ambiguous_examples[class_name].append(
                        {
                            "split": split,
                            "class": class_name,
                            "label_file": str(label_path.relative_to(ROOT_DIR)),
                            "image_file": str(image_path.relative_to(ROOT_DIR)) if image_path else "",
                            "annotation": line.strip(),
                        }
                    )
            for class_name in classes_in_image:
                image_counts[class_name] += 1

    print("\nContagem por classe:")
    for class_name in sorted(set(annotation_counts) | set(image_counts)):
        print(
            f"- {class_name}: {annotation_counts[class_name]} anotacoes, "
            f"{image_counts[class_name]} imagens"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = REPORTS_DIR / "class_format_comparison.json"
    comparison = {
        "yolo_classes": class_names,
        "coco_categories_by_split": coco_categories_by_split,
        "classes_in_both_formats": sorted(yolo_class_set & coco_class_set),
        "only_yolo": only_yolo,
        "only_coco": only_coco,
        "note": "Classes 0 e 1 exigem revisao manual antes de qualquer normalizacao.",
    }
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    examples_path = REPORTS_DIR / "ambiguous_class_examples.csv"
    with examples_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "class", "label_file", "image_file", "annotation"])
        writer.writeheader()
        for examples in ambiguous_examples.values():
            writer.writerows(examples)

    print(f"\nComparacao de classes salva em: {comparison_path}")
    print(f"Exemplos das classes 0 e 1 salvos em: {examples_path}")
    if not ambiguous_examples:
        print("Nenhuma anotacao das classes ambiguas 0 e 1 foi encontrada.")
    else:
        print("Analise manualmente esses exemplos antes de decidir se 0 ou 1 devem ser usados.")


if __name__ == "__main__":
    main()
