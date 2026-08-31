from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
COMPARISON_CSV = ROOT_DIR / "results" / "metrics" / "model_comparison.csv"
CHARTS_DIR = ROOT_DIR / "results" / "charts"


CHARTS = {
    "precision": ("precision_comparison.png", "Comparacao de precisao", "Precision"),
    "recall": ("recall_comparison.png", "Comparacao de recall", "Recall"),
    "f1_score": ("f1_score_comparison.png", "Comparacao de F1-score", "F1-score"),
    "map50": ("map50_comparison.png", "Comparacao de mAP@50", "mAP@50"),
    "map50_95": ("map50_95_comparison.png", "Comparacao de mAP@50-95", "mAP@50-95"),
    "mean_inference_seconds": ("inference_time_comparison.png", "Tempo medio de inferencia", "Segundos por imagem"),
    "false_positives": ("false_positives_comparison.png", "Falsos positivos", "Quantidade"),
}


def load_rows() -> list[dict]:
    if not COMPARISON_CSV.exists():
        raise FileNotFoundError(f"Arquivo de comparacao nao encontrado: {COMPARISON_CSV}")
    with COMPARISON_CSV.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot_metric(rows: list[dict], metric: str, filename: str, title: str, ylabel: str) -> None:
    models = [row["model"] for row in rows]
    values = [float(row[metric]) for row in rows]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(models, values, color=["#2563eb", "#dc2626"])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Modelo")
    if metric not in {"mean_inference_seconds", "false_positives"}:
        plt.ylim(0, 1)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / filename, dpi=150)
    plt.close()


def main() -> None:
    rows = load_rows()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    for metric, (filename, title, ylabel) in CHARTS.items():
        plot_metric(rows, metric, filename, title, ylabel)
        print(f"Grafico salvo: {CHARTS_DIR / filename}")


if __name__ == "__main__":
    main()

