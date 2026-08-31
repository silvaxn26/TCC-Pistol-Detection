from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

from dataset_roots import (
    COCO_BASE_DIR,
    IMAGE_EXTENSIONS,
    PROCESSED_DIR,
    ROOT_DIR,
    SPLITS,
    YOLO_BASE_DIR,
    find_coco_annotation,
    find_coco_dataset_root,
    find_image_by_stem,
    find_yolo_dataset_root,
    format_detected_root,
)


CLASS_NAMES_TO_ANALYZE = ("0", "1", "Handgun", "pistol", "pistols")
MAX_EXAMPLES_PER_CLASS = 8
EXAMPLES_DIR = ROOT_DIR / "results" / "predictions" / "class_examples"
REPORT_PATH = ROOT_DIR / "results" / "reports" / "class_analysis.md"


def read_yolo_classes(yolo_root: Path) -> list[str]:
    data = yaml.safe_load((yolo_root / "data.yaml").read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=lambda value: int(value))]
    return list(names)


def yolo_to_xyxy(line: str, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    _, x_center, y_center, width, height = [float(value) for value in parts]
    x1 = int((x_center - width / 2) * image_width)
    y1 = int((y_center - height / 2) * image_height)
    x2 = int((x_center + width / 2) * image_width)
    y2 = int((y_center + height / 2) * image_height)
    return max(0, x1), max(0, y1), min(image_width - 1, x2), min(image_height - 1, y2)


def save_yolo_example(image_path: Path, label_line: str, class_name: str, output_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    height, width = image.shape[:2]
    box = yolo_to_xyxy(label_line, width, height)
    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_name, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), image)


def save_coco_example(image_path: Path, bbox: list[float], class_name: str, output_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    x, y, width, height = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + width), int(y + height)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 255), 2)
    cv2.putText(image, class_name, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), image)


def collect_yolo_examples(yolo_root: Path, class_names: list[str]) -> tuple[Counter[str], dict[str, list[Path]]]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[Path]] = defaultdict(list)
    for split in SPLITS:
        labels_dir = yolo_root / split / "labels"
        images_dir = yolo_root / split / "images"
        if not labels_dir.exists():
            continue
        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = find_image_by_stem(images_dir, label_path.stem)
            for line_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(float(parts[0]))
                class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"INVALID_{class_id}"
                counts[class_name] += 1
                if class_name in CLASS_NAMES_TO_ANALYZE and image_path and len(examples[class_name]) < MAX_EXAMPLES_PER_CLASS:
                    output_path = EXAMPLES_DIR / class_name / f"{split}_{label_path.stem}_{line_index}.jpg"
                    if save_yolo_example(image_path, line, class_name, output_path):
                        examples[class_name].append(output_path)
    return counts, examples


def collect_coco_counts(coco_root: Path | None) -> Counter[str]:
    counts: Counter[str] = Counter()
    if coco_root is None:
        return counts
    for split in SPLITS:
        annotation_file = find_coco_annotation(coco_root / split)
        if not annotation_file:
            continue
        data = json.loads(annotation_file.read_text(encoding="utf-8"))
        category_by_id = {category["id"]: category["name"] for category in data.get("categories", [])}
        for annotation in data.get("annotations", []):
            counts[category_by_id.get(annotation.get("category_id"), "UNKNOWN")] += 1
    return counts


def find_coco_image(coco_root: Path, split: str, file_name: str) -> Path | None:
    candidates = [
        coco_root / split / file_name,
        coco_root / split / "images" / file_name,
        coco_root / "train" / file_name,
        coco_root / "train" / "images" / file_name,
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def add_coco_examples(coco_root: Path | None, examples: dict[str, list[Path]]) -> None:
    if coco_root is None:
        return
    for split in SPLITS:
        annotation_file = find_coco_annotation(coco_root / split)
        if not annotation_file:
            continue
        data = json.loads(annotation_file.read_text(encoding="utf-8"))
        category_by_id = {category["id"]: category["name"] for category in data.get("categories", [])}
        image_by_id = {image["id"]: image for image in data.get("images", [])}
        for annotation in data.get("annotations", []):
            class_name = category_by_id.get(annotation.get("category_id"), "UNKNOWN")
            if class_name not in CLASS_NAMES_TO_ANALYZE or len(examples[class_name]) >= MAX_EXAMPLES_PER_CLASS:
                continue
            image_info = image_by_id.get(annotation.get("image_id"))
            if not image_info:
                continue
            image_path = find_coco_image(coco_root, split, image_info["file_name"])
            if not image_path:
                continue
            output_path = EXAMPLES_DIR / class_name / f"coco_{split}_{Path(image_info['file_name']).stem}_{annotation['id']}.jpg"
            if save_coco_example(image_path, annotation["bbox"], class_name, output_path):
                examples[class_name].append(output_path)


def recommendation_text(class_name: str) -> str:
    if class_name in {"Handgun", "pistol", "pistols"}:
        return "Recomendacao inicial: normalizar para pistol."
    if class_name in {"0", "1"}:
        return (
            "Recomendacao inicial: revisar manualmente os exemplos antes de mapear. "
            "No codigo, esta classe fica desativada por padrao."
        )
    return "Recomendacao inicial: nao usar sem revisao."


def write_report(yolo_counts: Counter[str], coco_counts: Counter[str], examples: dict[str, list[Path]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Analise de classes ambiguas",
        "",
        "Relatorio gerado a partir dos datasets baixados do Roboflow. Ele apoia a decisao manual de normalizacao das classes antes do treinamento.",
        "",
        "## Contagem de anotacoes por classe",
        "",
        "| Classe | YOLOv8 | COCO | Recomendacao |",
        "| --- | ---: | ---: | --- |",
    ]
    all_classes = sorted(set(CLASS_NAMES_TO_ANALYZE) | set(yolo_counts) | set(coco_counts))
    for class_name in all_classes:
        lines.append(
            f"| `{class_name}` | {yolo_counts[class_name]} | {coco_counts[class_name]} | "
            f"{recommendation_text(class_name)} |"
        )

    lines.extend(["", "## Exemplos visuais", ""])
    for class_name in CLASS_NAMES_TO_ANALYZE:
        lines.append(f"### Classe `{class_name}`")
        class_examples = examples.get(class_name, [])
        if not class_examples:
            lines.append("")
            lines.append("Nenhum exemplo visual salvo para esta classe.")
            lines.append("")
            continue
        for example_path in class_examples:
            relative_path = example_path.relative_to(ROOT_DIR).as_posix()
            markdown_image_path = Path("..") / "predictions" / "class_examples" / class_name / example_path.name
            lines.append(f"![{class_name}]({markdown_image_path.as_posix()})")
            lines.append(f"`{relative_path}`")
            lines.append("")
        lines.append("")

    lines.extend(
        [
            "## Recomendacao de normalizacao",
            "",
            "- `Handgun`, `pistol` e `pistols`: normalizar para `pistol`.",
            "- `0` e `1`: manter desativadas por padrao ate revisao manual dos exemplos visuais.",
            "- A decisao final sobre `0` e `1` deve ser registrada no TCC antes de alterar o dicionario de mapeamento dos normalizadores.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    yolo_root = find_yolo_dataset_root()
    coco_root = find_coco_dataset_root()
    print(format_detected_root("YOLOv8", YOLO_BASE_DIR, yolo_root))
    print(format_detected_root("COCO", COCO_BASE_DIR, coco_root))
    if yolo_root is None:
        print("Dataset YOLOv8 nao encontrado. Execute primeiro src/dataset/download_roboflow_yolo.py.")
        raise SystemExit(1)

    if EXAMPLES_DIR.exists():
        shutil.rmtree(EXAMPLES_DIR)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    class_names = read_yolo_classes(yolo_root)
    yolo_counts, examples = collect_yolo_examples(yolo_root, class_names)
    coco_counts = collect_coco_counts(coco_root)
    add_coco_examples(coco_root, examples)
    write_report(yolo_counts, coco_counts, examples)

    print(f"Exemplos visuais salvos em: {EXAMPLES_DIR}")
    print(f"Relatorio salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    main()
