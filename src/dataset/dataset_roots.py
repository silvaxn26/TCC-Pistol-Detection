from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
YOLO_BASE_DIR = ROOT_DIR / "data" / "roboflow_yolo"
COCO_BASE_DIR = ROOT_DIR / "data" / "roboflow_coco"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_YOLO_DIR = PROCESSED_DIR / "yolo"
PROCESSED_COCO_DIR = PROCESSED_DIR / "coco"
SPLITS = ("train", "valid", "test")
COCO_ANNOTATION_NAMES = ("_annotations.coco.json", "_annotations.json", "annotations.json")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def has_split_dirs(dataset_root: Path) -> bool:
    return all((dataset_root / split).exists() for split in SPLITS)


def find_coco_annotation(split_dir: Path) -> Path | None:
    for name in COCO_ANNOTATION_NAMES:
        candidate = split_dir / name
        if candidate.exists():
            return candidate
    matches = sorted(split_dir.glob("*.json"))
    return matches[0] if matches else None


def find_image_by_stem(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def find_yolo_dataset_root(base_dir: Path = YOLO_BASE_DIR) -> Path | None:
    if not base_dir.exists():
        return None

    candidates = sorted(base_dir.rglob("data.yaml"))
    if not candidates:
        return None

    roots = [candidate.parent for candidate in candidates]
    for root in roots:
        if has_split_dirs(root):
            return root
    return roots[0]


def find_coco_dataset_root(base_dir: Path = COCO_BASE_DIR) -> Path | None:
    if not base_dir.exists():
        return None

    annotation_files = []
    for name in COCO_ANNOTATION_NAMES:
        annotation_files.extend(base_dir.rglob(name))
    if not annotation_files:
        annotation_files = list(base_dir.rglob("*.json"))
    if not annotation_files:
        return None

    candidate_roots = []
    for annotation_file in sorted(set(annotation_files)):
        parent = annotation_file.parent
        candidate_roots.append(parent.parent if parent.name in SPLITS else parent)

    for root in candidate_roots:
        if has_split_dirs(root):
            return root
    return candidate_roots[0]


def format_detected_root(label: str, base_dir: Path, dataset_root: Path | None) -> str:
    if dataset_root is None:
        return f"{label}: raiz real nao detectada dentro de {base_dir}"
    return f"{label}: raiz real detectada em {dataset_root}"
