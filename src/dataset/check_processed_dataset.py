from __future__ import annotations

import json
from collections import Counter
import yaml

from dataset_roots import PROCESSED_COCO_DIR, PROCESSED_YOLO_DIR, SPLITS, find_coco_annotation, list_images


def check_yolo() -> tuple[list[str], dict[str, dict[str, int]]]:
    problems = []
    summary = {}
    data_yaml_path = PROCESSED_YOLO_DIR / "data.yaml"
    if not data_yaml_path.exists():
        return [f"Arquivo ausente: {data_yaml_path}"], summary

    data_yaml = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    names = data_yaml.get("names", [])
    if names != ["pistol"]:
        problems.append(f"Classes YOLO finais invalidas: {names}. Esperado: ['pistol']")
    if data_yaml.get("nc") != 1:
        problems.append(f"Valor nc invalido no YOLO: {data_yaml.get('nc')}. Esperado: 1")

    for split in SPLITS:
        images_dir = PROCESSED_YOLO_DIR / split / "images"
        labels_dir = PROCESSED_YOLO_DIR / split / "labels"
        if not images_dir.exists():
            problems.append(f"Pasta ausente: {images_dir}")
            continue
        if not labels_dir.exists():
            problems.append(f"Pasta ausente: {labels_dir}")
            continue

        images = list_images(images_dir)
        labels = sorted(labels_dir.glob("*.txt"))
        image_stems = {image.stem for image in images}
        annotation_count = 0
        for image in images:
            label_path = labels_dir / f"{image.stem}.txt"
            if not label_path.exists():
                problems.append(f"Imagem sem label correspondente: {image}")
        for label_path in labels:
            if label_path.stem not in image_stems:
                problems.append(f"Label sem imagem correspondente: {label_path}")
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                annotation_count += 1
                if len(parts) != 5:
                    problems.append(f"Label YOLO invalido em {label_path}:{line_number}")
                    continue
                if parts[0] != "0":
                    problems.append(f"Classe YOLO final invalida em {label_path}:{line_number}: {parts[0]}")
        summary[split] = {"images": len(images), "labels": len(labels), "annotations": annotation_count}
    return problems, summary


def check_coco() -> tuple[list[str], dict[str, dict[str, int]]]:
    problems = []
    summary = {}
    if not PROCESSED_COCO_DIR.exists():
        return [f"Pasta ausente: {PROCESSED_COCO_DIR}"], summary

    for split in SPLITS:
        split_dir = PROCESSED_COCO_DIR / split
        annotation_file = find_coco_annotation(split_dir)
        if not split_dir.exists():
            problems.append(f"Split COCO ausente: {split_dir}")
            continue
        if not annotation_file:
            problems.append(f"Arquivo _annotations.coco.json ausente em: {split_dir}")
            continue

        data = json.loads(annotation_file.read_text(encoding="utf-8"))
        categories = data.get("categories", [])
        if categories != [{"id": 1, "name": "pistol", "supercategory": "weapon"}]:
            problems.append(f"Categorias COCO invalidas em {annotation_file}: {categories}")
        category_counts = Counter(annotation.get("category_id") for annotation in data.get("annotations", []))
        invalid_categories = sorted(category_id for category_id in category_counts if category_id != 1)
        if invalid_categories:
            problems.append(f"Annotations COCO com category_id invalido em {annotation_file}: {invalid_categories}")
        summary[split] = {
            "images": len(data.get("images", [])),
            "annotations": len(data.get("annotations", [])),
        }
    return problems, summary


def main() -> None:
    print("Validando dataset processado final")
    print(f"YOLO: {PROCESSED_YOLO_DIR}")
    print(f"COCO: {PROCESSED_COCO_DIR}")

    yolo_problems, yolo_summary = check_yolo()
    coco_problems, coco_summary = check_coco()
    problems = yolo_problems + coco_problems

    print("\nResumo YOLO:")
    for split, values in yolo_summary.items():
        print(f"- {split}: {values['images']} imagens, {values['labels']} labels, {values['annotations']} anotacoes")

    print("\nResumo COCO:")
    for split, values in coco_summary.items():
        print(f"- {split}: {values['images']} imagens, {values['annotations']} anotacoes")

    if problems:
        print("\nProblemas encontrados:")
        for problem in problems[:80]:
            print(f"- {problem}")
        if len(problems) > 80:
            print(f"... mais {len(problems) - 80} problemas omitidos")
        raise SystemExit(1)

    print("\nDataset processado validado com sucesso. Classes finais: ['pistol']")


if __name__ == "__main__":
    main()
