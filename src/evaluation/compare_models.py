from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT_DIR / "results" / "metrics"
REPORT_PATH = ROOT_DIR / "results" / "reports" / "model_comparison_report.md"
COMPARISON_JSON = METRICS_DIR / "model_comparison.json"
COMPARISON_CSV = METRICS_DIR / "model_comparison.csv"


METRICS = [
    "precision",
    "recall",
    "f1_score",
    "map50",
    "map50_95",
    "mean_inference_seconds",
    "false_positives",
]


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de metricas nao encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def best_model(rows: list[dict], metric: str, lower_is_better: bool = False) -> str:
    key = (lambda row: row[metric])
    selected = min(rows, key=key) if lower_is_better else max(rows, key=key)
    return selected["model"]


def write_csv(rows: list[dict]) -> None:
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", *METRICS])
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict], winners: dict[str, str]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Comparacao entre YOLOv8 Nano e Faster R-CNN",
        "",
        "Relatorio gerado a partir dos arquivos de metricas produzidos pelos scripts de avaliacao.",
        "",
        "## Tabela comparativa",
        "",
        "| Modelo | Precision | Recall | F1-score | mAP@50 | mAP@50-95 | Tempo medio (s) | Falsos positivos |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['f1_score']:.6f} | {row['map50']:.6f} | {row['map50_95']:.6f} | "
            f"{row['mean_inference_seconds']:.6f} | {int(row['false_positives'])} |"
        )

    lines.extend(
        [
            "",
            "## Melhores resultados por criterio",
            "",
            f"- Melhor precisao: **{winners['precision']}**",
            f"- Melhor recall: **{winners['recall']}**",
            f"- Melhor F1-score: **{winners['f1_score']}**",
            f"- Melhor mAP@50: **{winners['map50']}**",
            f"- Melhor mAP@50-95: **{winners['map50_95']}**",
            f"- Modelo mais rapido: **{winners['mean_inference_seconds']}**",
            f"- Menor quantidade de falsos positivos: **{winners['false_positives']}**",
            "",
            "As interpretacoes academicas devem considerar tambem exemplos qualitativos, qualidade das imagens, oclusoes e objetos semelhantes.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    yolo = load_metrics(METRICS_DIR / "yolo_metrics.json")
    faster = load_metrics(METRICS_DIR / "faster_rcnn_metrics.json")
    rows = [
        {"model": "YOLOv8 Nano", **{metric: float(yolo[metric]) for metric in METRICS}},
        {"model": "Faster R-CNN", **{metric: float(faster[metric]) for metric in METRICS}},
    ]

    winners = {
        "precision": best_model(rows, "precision"),
        "recall": best_model(rows, "recall"),
        "f1_score": best_model(rows, "f1_score"),
        "map50": best_model(rows, "map50"),
        "map50_95": best_model(rows, "map50_95"),
        "mean_inference_seconds": best_model(rows, "mean_inference_seconds", lower_is_better=True),
        "false_positives": best_model(rows, "false_positives", lower_is_better=True),
    }
    payload = {"models": rows, "winners": winners}

    write_csv(rows)
    COMPARISON_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(rows, winners)
    print(f"Comparacao CSV salva em: {COMPARISON_CSV}")
    print(f"Comparacao JSON salva em: {COMPARISON_JSON}")
    print(f"Relatorio salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    main()
