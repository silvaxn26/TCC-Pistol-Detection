from __future__ import annotations

import json
from collections import Counter

from dataset_roots import PROCESSED_COCO_DIR, ROOT_DIR, SPLITS, find_coco_annotation


# Ajuste este dicionario somente apos analisar results/reports/class_analysis.md.
# Use "pistol" para manter/mapear a classe e None para ignorar a classe.
CLASS_MAPPING: dict[str, str | None] = {
    "Handgun": "pistol",
    "pistol": "pistol",
    "pistols": "pistol",
    "0": None,
    "1": None,
}

REPORT_PATH = ROOT_DIR / "results" / "reports" / "coco_normalization_report.json"


def normalize_file(annotation_file):
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    category_by_id = {category["id"]: category["name"] for category in data.get("categories", [])}
    kept_annotations = []
    kept_image_ids = set()
    skipped_by_class: Counter[str] = Counter()

    for annotation in data.get("annotations", []):
        class_name = category_by_id.get(annotation.get("category_id"), "UNKNOWN")
        mapped_class = CLASS_MAPPING.get(class_name)
        if mapped_class == "pistol":
            new_annotation = dict(annotation)
            new_annotation["category_id"] = 1
            kept_annotations.append(new_annotation)
            kept_image_ids.add(annotation["image_id"])
        else:
            skipped_by_class[class_name] += 1

    normalized = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": [image for image in data.get("images", []) if image.get("id") in kept_image_ids],
        "annotations": kept_annotations,
        "categories": [{"id": 1, "name": "pistol", "supercategory": "weapon"}],
    }
    annotation_file.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(normalized["images"]), len(kept_annotations), skipped_by_class


def main() -> None:
    if not PROCESSED_COCO_DIR.exists():
        raise FileNotFoundError(
            f"Dataset COCO processado nao encontrado: {PROCESSED_COCO_DIR}. "
            "Execute primeiro src/dataset/split_coco_dataset.py."
        )

    print(f"Dataset COCO processado: {PROCESSED_COCO_DIR}")
    print(f"Mapeamento configurado: {CLASS_MAPPING}")

    summary = {}
    total_images = 0
    total_annotations = 0
    skipped_by_class: Counter[str] = Counter()

    for split in SPLITS:
        annotation_file = find_coco_annotation(PROCESSED_COCO_DIR / split)
        if not annotation_file:
            print(f"Aviso: anotacao COCO ausente no split {split}")
            continue
        image_count, annotation_count, skipped_counter = normalize_file(annotation_file)
        total_images += image_count
        total_annotations += annotation_count
        skipped_by_class.update(skipped_counter)
        summary[split] = {"images": image_count, "annotations": annotation_count}
        print(f"{split}: {image_count} imagens, {annotation_count} anotacoes normalizadas")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "processed_root": str(PROCESSED_COCO_DIR),
        "class_mapping": CLASS_MAPPING,
        "total_images": total_images,
        "total_annotations": total_annotations,
        "skipped_by_class": dict(skipped_by_class),
        "splits": summary,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Categorias finais: [{'id': 1, 'name': 'pistol'}]")
    print(f"Relatorio salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    main()

