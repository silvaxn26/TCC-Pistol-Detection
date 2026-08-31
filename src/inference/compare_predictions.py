from __future__ import annotations

import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = ROOT_DIR / "results" / "predictions"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    yolo = load_json(PREDICTIONS_DIR / "yolo_prediction_summary.json")
    faster = load_json(PREDICTIONS_DIR / "faster_rcnn_prediction.json")
    comparison = {
        "yolo_mean_seconds_per_image": yolo.get("mean_seconds_per_image"),
        "faster_rcnn_seconds_single_image": faster.get("seconds"),
        "yolo_images": yolo.get("images"),
        "faster_rcnn_detections": len(faster.get("detections", [])) if faster else None,
    }
    output_path = PREDICTIONS_DIR / "prediction_comparison.json"
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

