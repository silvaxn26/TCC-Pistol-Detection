from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from dataset_roots import (
    COCO_BASE_DIR,
    PROCESSED_COCO_DIR,
    ROOT_DIR,
    SPLITS,
    find_coco_annotation,
    find_coco_dataset_root,
    format_detected_root,
)


RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10
YOLO_MANIFEST_PATH = ROOT_DIR / "results" / "reports" / "yolo_split_manifest.json"


def load_yolo_split_by_filename() -> dict[str, str]:
    if not YOLO_MANIFEST_PATH.exists():
        return {}
    manifest = json.loads(YOLO_MANIFEST_PATH.read_text(encoding="utf-8"))
    mapping = {}
    for split, filenames in manifest.get("splits", {}).items():
        for filename in filenames:
            mapping[Path(filename).name] = split
            mapping[Path(filename).stem] = split
    return mapping


def split_names_random(image_names: list[str]) -> dict[str, str]:
    shuffled = image_names[:]
    random.Random(RANDOM_SEED).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)
    mapping = {}
    for name in shuffled[:train_end]:
        mapping[name] = "train"
    for name in shuffled[train_end:valid_end]:
        mapping[name] = "valid"
    for name in shuffled[valid_end:]:
        mapping[name] = "test"
    return mapping


def load_all_coco(source_root: Path) -> tuple[dict, dict[int, str]]:
    merged = {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": []}
    source_split_by_image_id = {}
    seen_images = set()
    seen_annotations = set()
    categories_by_id = {}

    for split in SPLITS:
        split_dir = source_root / split
        annotation_file = find_coco_annotation(split_dir)
        if not annotation_file:
            continue
        data = json.loads(annotation_file.read_text(encoding="utf-8"))
        merged["info"] = data.get("info", merged["info"])
        merged["licenses"] = data.get("licenses", merged["licenses"])
        for category in data.get("categories", []):
            categories_by_id[category["id"]] = category
        for image in data.get("images", []):
            if image["id"] not in seen_images:
                merged["images"].append(image)
                seen_images.add(image["id"])
            source_split_by_image_id[image["id"]] = split
        for annotation in data.get("annotations", []):
            if annotation["id"] not in seen_annotations:
                merged["annotations"].append(annotation)
                seen_annotations.add(annotation["id"])

    merged["categories"] = list(categories_by_id.values())
    return merged, source_split_by_image_id


def find_source_image(source_root: Path, source_split: str, file_name: str) -> Path | None:
    candidates = [
        source_root / source_split / file_name,
        source_root / source_split / "images" / file_name,
        source_root / "train" / file_name,
        source_root / "train" / "images" / file_name,
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def write_split(
    source_root: Path,
    data: dict,
    source_split_by_image_id: dict[int, str],
    target_split: str,
    image_ids: set[int],
) -> tuple[int, int]:
    target_dir = PROCESSED_COCO_DIR / target_split
    target_dir.mkdir(parents=True, exist_ok=True)
    images = [image for image in data["images"] if image["id"] in image_ids]
    annotations = [annotation for annotation in data["annotations"] if annotation["image_id"] in image_ids]

    for image in images:
        source_split = source_split_by_image_id.get(image["id"], "train")
        source_image = find_source_image(source_root, source_split, image["file_name"])
        if source_image:
            target_image = target_dir / image["file_name"]
            target_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, target_image)

    split_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": data.get("categories", []),
    }
    (target_dir / "_annotations.coco.json").write_text(
        json.dumps(split_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(images), len(annotations)


def main() -> None:
    source_root = find_coco_dataset_root()
    print(format_detected_root("COCO", COCO_BASE_DIR, source_root))
    if source_root is None:
        print("Dataset COCO nao encontrado. Execute primeiro src/dataset/download_roboflow_coco.py.")
        raise SystemExit(1)

    data, source_split_by_image_id = load_all_coco(source_root)
    if not data["images"]:
        raise ValueError("Nenhuma imagem encontrada nas anotacoes COCO.")

    yolo_split_by_name = load_yolo_split_by_filename()
    if yolo_split_by_name:
        print(f"Usando split do YOLO salvo em: {YOLO_MANIFEST_PATH}")
    else:
        print("Manifesto YOLO nao encontrado. Criando split COCO aleatorio com seed fixa.")

    image_names = [Path(image["file_name"]).name for image in data["images"]]
    fallback_split = split_names_random(image_names)
    image_ids_by_split = {split: set() for split in SPLITS}

    for image in data["images"]:
        basename = Path(image["file_name"]).name
        stem = Path(image["file_name"]).stem
        split = yolo_split_by_name.get(basename) or yolo_split_by_name.get(stem) or fallback_split[basename]
        image_ids_by_split[split].add(image["id"])

    if PROCESSED_COCO_DIR.exists():
        shutil.rmtree(PROCESSED_COCO_DIR)
    PROCESSED_COCO_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        image_count, annotation_count = write_split(source_root, data, source_split_by_image_id, split, image_ids_by_split[split])
        print(f"{split}: {image_count} imagens, {annotation_count} anotacoes")

    print(f"COCO dividido em: {PROCESSED_COCO_DIR}")


if __name__ == "__main__":
    main()

