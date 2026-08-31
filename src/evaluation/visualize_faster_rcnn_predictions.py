from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "training_config.yaml"
MODEL_PATH = ROOT_DIR / "models" / "faster_rcnn" / "best.pt"
OUTPUT_DIR = ROOT_DIR / "results" / "predictions" / "faster_rcnn"
MAX_IMAGES = 20
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.25


def build_model(num_classes: int = 2):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(device: torch.device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = build_model(num_classes=int(checkpoint.get("num_classes", 2)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def find_image(split_dir: Path, file_name: str) -> Path:
    candidates = [split_dir / file_name, split_dir / "images" / file_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Imagem nao encontrada: {file_name}")


@torch.no_grad()
def draw_prediction(model, image_path: Path, output_path: Path, device: torch.device) -> None:
    image = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(image).to(device)
    prediction = model([tensor])[0]

    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)
    for box, score, label in zip(prediction["boxes"].cpu(), prediction["scores"].cpu(), prediction["labels"].cpu()):
        if int(label) != 1 or float(score) < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        draw.text((x1, max(0, y1 - 16)), f"pistol {float(score):.2f}", fill="lime")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo Faster R-CNN nao encontrado: {MODEL_PATH}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["faster_rcnn"]
    split_dir = ROOT_DIR / config["coco_dir"] / "test"
    data = json.loads((split_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    images = data["images"][:]
    random.Random(RANDOM_SEED).shuffle(images)
    images = images[: min(MAX_IMAGES, len(images))]

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Gerando predicoes Faster R-CNN em: {device}")
    model = load_model(device)

    for image_info in images:
        image_path = find_image(split_dir, image_info["file_name"])
        output_path = OUTPUT_DIR / Path(image_info["file_name"]).name
        draw_prediction(model, image_path, output_path, device)

    print(f"Predicoes salvas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

