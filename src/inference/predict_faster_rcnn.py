from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT_DIR / "models" / "faster_rcnn" / "faster_rcnn_pistol.pth"
PREDICTIONS_DIR = ROOT_DIR / "results" / "predictions"


def build_model(num_classes: int = 2):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia com Faster R-CNN.")
    parser.add_argument("--image", required=True, help="Caminho da imagem.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Caminho do checkpoint .pth.")
    parser.add_argument("--conf", type=float, default=0.5, help="Limiar minimo de confianca.")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(image).to(device)

    start = perf_counter()
    with torch.no_grad():
        prediction = model([tensor])[0]
    elapsed = perf_counter() - start

    keep = prediction["scores"] >= args.conf
    output = {
        "image": str(image_path),
        "model": str(model_path),
        "seconds": elapsed,
        "detections": [
            {
                "label": "pistol",
                "score": float(score),
                "box": [float(value) for value in box],
            }
            for box, score in zip(prediction["boxes"][keep].cpu(), prediction["scores"][keep].cpu())
        ],
    }
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTIONS_DIR / "faster_rcnn_prediction.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

