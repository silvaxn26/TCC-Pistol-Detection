from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import yaml
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "training_config.yaml"
PREDICTIONS_DIR = ROOT_DIR / "results" / "predictions"


def default_model_path() -> Path:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["yolo"]
    return ROOT_DIR / config["project"] / config["name"] / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia com YOLOv8.")
    parser.add_argument("--source", required=True, help="Caminho de uma imagem ou pasta de imagens.")
    parser.add_argument("--model", default=str(default_model_path()), help="Caminho do arquivo .pt treinado.")
    parser.add_argument("--conf", type=float, default=0.25, help="Limiar minimo de confianca.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

    model = YOLO(str(model_path))
    start = perf_counter()
    results = model.predict(source=args.source, conf=args.conf, save=True, project=str(PREDICTIONS_DIR), name="yolo")
    elapsed = perf_counter() - start

    summary = {
        "model": str(model_path),
        "source": args.source,
        "images": len(results),
        "total_seconds": elapsed,
        "mean_seconds_per_image": elapsed / len(results) if results else 0.0,
    }
    output_path = PREDICTIONS_DIR / "yolo_prediction_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

