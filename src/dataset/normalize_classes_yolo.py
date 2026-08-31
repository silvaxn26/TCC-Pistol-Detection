from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

from dataset_roots import PROCESSED_YOLO_DIR, ROOT_DIR, SPLITS, find_image_by_stem


# Ajuste este dicionario somente apos analisar results/reports/class_analysis.md.
# Use "pistol" para manter/mapear a classe e None para ignorar a classe.
CLASS_MAPPING: dict[str, str | None] = {
    "Handgun": "pistol",
    "pistol": "pistol",
    "pistols": "pistol",
    "0": None,
    "1": None,
}

REPORT_PATH = ROOT_DIR / "results" / "reports" / "yolo_normalization_report.json"


def read_class_names() -> list[str]:
    data_yaml_path = PROCESSED_YOLO_DIR / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {data_yaml_path}. "
            "Execute primeiro src/dataset/split_yolo_dataset.py."
        )
    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=lambda value: int(value))]
    return list(names)


def normalize_label_file(label_path: Path, class_names: list[str]) -> tuple[int, int, Counter[str]]:
    output_lines = []
    kept = 0
    skipped = 0
    skipped_by_class: Counter[str] = Counter()

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            skipped += 1
            skipped_by_class["INVALID_LINE"] += 1
            continue
        class_id = int(float(parts[0]))
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"INVALID_{class_id}"
        mapped_class = CLASS_MAPPING.get(class_name)
        if mapped_class == "pistol":
            output_lines.append("0 " + " ".join(parts[1:]))
            kept += 1
        else:
            skipped += 1
            skipped_by_class[class_name] += 1

    label_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    return kept, skipped, skipped_by_class


def main() -> None:
    class_names = read_class_names()
    print(f"Dataset YOLO processado: {PROCESSED_YOLO_DIR}")
    print(f"Classes antes da normalizacao: {class_names}")
    print(f"Mapeamento configurado: {CLASS_MAPPING}")

    total_kept = 0
    total_skipped = 0
    skipped_by_class: Counter[str] = Counter()
    split_summary = {}

    for split in SPLITS:
        labels_dir = PROCESSED_YOLO_DIR / split / "labels"
        if not labels_dir.exists():
            print(f"Aviso: split YOLO ausente: {labels_dir}")
            continue
        split_kept = 0
        split_skipped = 0
        for label_path in sorted(labels_dir.glob("*.txt")):
            kept, skipped, skipped_counter = normalize_label_file(label_path, class_names)
            if kept == 0:
                image_path = find_image_by_stem(PROCESSED_YOLO_DIR / split / "images", label_path.stem)
                if image_path:
                    image_path.unlink()
                label_path.unlink()
            split_kept += kept
            split_skipped += skipped
            skipped_by_class.update(skipped_counter)
        total_kept += split_kept
        total_skipped += split_skipped
        split_summary[split] = {"kept_annotations": split_kept, "skipped_annotations": split_skipped}
        print(f"{split}: {split_kept} anotacoes mantidas, {split_skipped} ignoradas")

    data_yaml = {
        "path": str(PROCESSED_YOLO_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": ["pistol"],
        "nc": 1,
    }
    (PROCESSED_YOLO_DIR / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "processed_root": str(PROCESSED_YOLO_DIR),
        "class_mapping": CLASS_MAPPING,
        "total_kept_annotations": total_kept,
        "total_skipped_annotations": total_skipped,
        "skipped_by_class": dict(skipped_by_class),
        "splits": split_summary,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Novo data.yaml gerado com names: ['pistol'] e nc: 1")
    print(f"Relatorio salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    main()
