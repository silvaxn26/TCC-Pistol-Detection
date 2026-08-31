from __future__ import annotations

from pathlib import Path

import yaml

from dataset_roots import YOLO_BASE_DIR, find_yolo_dataset_root, format_detected_root


ROOT_DIR = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


def load_data_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo data.yaml nao encontrado em {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def read_class_names(data_yaml: dict) -> list[str]:
    names = data_yaml.get("names", [])
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=lambda value: int(value))]
    return list(names)


def validate_label_file(label_path: Path, class_count: int) -> list[str]:
    errors = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_number} deve conter 5 valores")
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_number} contem valor nao numerico")
            continue
        if class_id < 0 or class_id >= class_count:
            errors.append(f"{label_path}:{line_number} class_id {class_id} fora de 0..{class_count - 1}")
        if any(value < 0 or value > 1 for value in values):
            errors.append(f"{label_path}:{line_number} coordenadas YOLO devem estar entre 0 e 1")
    return errors


def main() -> None:
    yolo_dir = find_yolo_dataset_root()
    print(format_detected_root("YOLOv8", YOLO_BASE_DIR, yolo_dir))
    if yolo_dir is None:
        print("Dataset YOLOv8 nao encontrado. Execute primeiro src/dataset/download_roboflow_yolo.py.")
        raise SystemExit(1)

    data_yaml_path = yolo_dir / "data.yaml"
    data_yaml = load_data_yaml(data_yaml_path)
    class_names = read_class_names(data_yaml)
    if not class_names:
        raise ValueError("Nenhuma classe encontrada em data.yaml")

    print(f"Dataset YOLO: {yolo_dir}")
    print(f"Classes ({len(class_names)}): {class_names}")

    all_errors = []
    for split in SPLITS:
        images_dir = yolo_dir / split / "images"
        labels_dir = yolo_dir / split / "labels"
        if not images_dir.exists():
            all_errors.append(f"Pasta ausente: {images_dir}")
            continue
        if not labels_dir.exists():
            all_errors.append(f"Pasta ausente: {labels_dir}")
            continue

        images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        labels = sorted(labels_dir.glob("*.txt"))
        missing_labels = [image.name for image in images if not (labels_dir / f"{image.stem}.txt").exists()]
        orphan_labels = [label.name for label in labels if not any((images_dir / f"{label.stem}{ext}").exists() for ext in IMAGE_EXTENSIONS)]

        for label in labels:
            all_errors.extend(validate_label_file(label, len(class_names)))

        print(
            f"{split}: {len(images)} imagens, {len(labels)} labels, "
            f"{len(missing_labels)} imagens sem label, {len(orphan_labels)} labels sem imagem"
        )
        if missing_labels[:5]:
            print(f"  Exemplos sem label: {missing_labels[:5]}")
        if orphan_labels[:5]:
            print(f"  Exemplos sem imagem: {orphan_labels[:5]}")

    if all_errors:
        print("\nProblemas encontrados:")
        for error in all_errors[:50]:
            print(f"- {error}")
        if len(all_errors) > 50:
            print(f"... mais {len(all_errors) - 50} problemas omitidos")
        raise SystemExit(1)

    print("\nValidacao concluida sem erros estruturais.")


if __name__ == "__main__":
    main()
