from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import yaml

from dataset_roots import (
    PROCESSED_YOLO_DIR,
    ROOT_DIR,
    SPLITS,
    YOLO_BASE_DIR,
    find_yolo_dataset_root,
    format_detected_root,
    list_images,
)


RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10
MANIFEST_PATH = ROOT_DIR / "results" / "reports" / "yolo_split_manifest.json"


def split_items(items: list[Path]) -> dict[str, list[Path]]:
    shuffled = items[:]
    random.Random(RANDOM_SEED).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def existing_split_images(source_root: Path) -> dict[str, list[Path]]:
    return {split: list_images(source_root / split / "images") for split in SPLITS}


def choose_split(source_root: Path) -> dict[str, list[Path]]:
    images_by_split = existing_split_images(source_root)
    if all(images_by_split[split] for split in SPLITS):
        return images_by_split
    train_images = images_by_split["train"]
    if not train_images:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {source_root / 'train' / 'images'}")
    print("YOLOv8 possui split incompleto. Criando train/valid/test a partir do train original.")
    return split_items(train_images)


def copy_pair(image_path: Path, source_root: Path, target_split: str) -> str | None:
    source_labels_dir = image_path.parent.parent / "labels"
    label_path = source_labels_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        print(f"Aviso: label ausente para imagem {image_path}")
        return None

    target_images = PROCESSED_YOLO_DIR / target_split / "images"
    target_labels = PROCESSED_YOLO_DIR / target_split / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, target_images / image_path.name)
    shutil.copy2(label_path, target_labels / label_path.name)
    return image_path.name


def write_data_yaml(source_root: Path) -> None:
    source_yaml = yaml.safe_load((source_root / "data.yaml").read_text(encoding="utf-8")) or {}
    data_yaml = {
        "path": str(PROCESSED_YOLO_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": source_yaml.get("names", []),
        "nc": source_yaml.get("nc", len(source_yaml.get("names", []))),
    }
    (PROCESSED_YOLO_DIR / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")


def main() -> None:
    source_root = find_yolo_dataset_root()
    print(format_detected_root("YOLOv8", YOLO_BASE_DIR, source_root))
    if source_root is None:
        print("Dataset YOLOv8 nao encontrado. Execute primeiro src/dataset/download_roboflow_yolo.py.")
        raise SystemExit(1)

    if PROCESSED_YOLO_DIR.exists():
        shutil.rmtree(PROCESSED_YOLO_DIR)
    PROCESSED_YOLO_DIR.mkdir(parents=True, exist_ok=True)

    split_map = choose_split(source_root)
    manifest = {
        "source_root": str(source_root),
        "processed_root": str(PROCESSED_YOLO_DIR),
        "seed": RANDOM_SEED,
        "ratios": {"train": TRAIN_RATIO, "valid": VALID_RATIO, "test": TEST_RATIO},
        "splits": {},
    }

    for split, images in split_map.items():
        copied_names = []
        for image_path in images:
            copied = copy_pair(image_path, source_root, split)
            if copied:
                copied_names.append(copied)
        manifest["splits"][split] = copied_names
        print(f"{split}: {len(copied_names)} imagens copiadas")

    write_data_yaml(source_root)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"YOLOv8 dividido em: {PROCESSED_YOLO_DIR}")
    print(f"Manifesto de split salvo em: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
