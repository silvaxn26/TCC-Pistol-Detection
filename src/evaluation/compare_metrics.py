from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT_DIR / "results" / "metrics"
CHARTS_DIR = ROOT_DIR / "results" / "charts"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    yolo = load_json(METRICS_DIR / "yolo_metrics.json")
    faster = load_json(METRICS_DIR / "faster_rcnn_metrics.json")
    rows = []
    if yolo:
        rows.append({"model": "YOLOv8 Nano", **yolo})
    if faster:
        rows.append({"model": "Faster R-CNN", **faster})

    if not rows:
        print("Nenhum arquivo de metricas encontrado em results/metrics.")
        return

    df = pd.DataFrame(rows)
    output_path = METRICS_DIR / "metrics_comparison.csv"
    df.to_csv(output_path, index=False)

    metric_columns = [column for column in ["precision", "recall", "f1_score", "map50", "map50_95"] if column in df]
    if metric_columns:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        chart_path = CHARTS_DIR / "metrics_comparison.png"
        df.set_index("model")[metric_columns].plot(kind="bar", figsize=(10, 5))
        plt.ylim(0, 1)
        plt.ylabel("Valor")
        plt.title("Comparacao de metricas")
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Grafico salvo em: {chart_path}")

    print(df.to_string(index=False))
    print(f"Comparacao salva em: {output_path}")


if __name__ == "__main__":
    main()
