from __future__ import annotations

from pathlib import Path

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


def check_yolo(yolo_root: Path | None) -> list[str]:
    problems = []
    if yolo_root is None:
        return [f"Raiz YOLOv8 nao detectada. Baixe o dataset em {YOLO_BASE_DIR}."]
    if not (yolo_root / "data.yaml").exists():
        problems.append(f"Arquivo YOLOv8 ausente: {yolo_root / 'data.yaml'}")
    for split in SPLITS:
        split_dir = yolo_root / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not split_dir.exists():
            problems.append(f"Split YOLOv8 ausente: {split_dir}")
            continue
        if not images_dir.exists():
            problems.append(f"Pasta de imagens YOLOv8 ausente: {images_dir}")
        if not labels_dir.exists():
            problems.append(f"Pasta de labels YOLOv8 ausente: {labels_dir}")
    return problems


def check_coco(coco_root: Path | None) -> list[str]:
    problems = []
    if coco_root is None:
        return [f"Raiz COCO nao detectada. Baixe o dataset em {COCO_BASE_DIR}."]
    for split in SPLITS:
        split_dir = coco_root / split
        if not split_dir.exists():
            problems.append(f"Split COCO ausente: {split_dir}")
            continue
        if not find_coco_annotation(split_dir):
            problems.append(f"Arquivo _annotations.coco.json ausente no split COCO: {split_dir}")
    return problems


def main() -> None:
    print("Verificando datasets baixados do Roboflow")
    print("Versao esperada: Workspace=silvaxn Project=Pistol_detection Version=1")
    yolo_root = find_yolo_dataset_root()
    coco_root = find_coco_dataset_root()
    print(format_detected_root("YOLOv8", YOLO_BASE_DIR, yolo_root))
    print(format_detected_root("COCO", COCO_BASE_DIR, coco_root))

    problems = check_yolo(yolo_root) + check_coco(coco_root)
    if problems:
        print("\nProblemas encontrados:")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)

    print("\nEstrutura encontrada:")
    for split in SPLITS:
        print(f"- YOLOv8 {split}: {yolo_root / split}")
        print(f"- COCO {split}: {coco_root / split}")
    print("\nVerificacao concluida. As duas pastas existem e possuem train, valid e test.")


if __name__ == "__main__":
    main()
