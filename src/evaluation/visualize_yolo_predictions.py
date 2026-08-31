from __future__ import annotations

import random
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "training_config.yaml"
OUTPUT_DIR = ROOT_DIR / "results" / "predictions" / "yolo"
BEST_MODEL_PATH = ROOT_DIR / "models" / "yolo" / "best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGES = 20
RANDOM_SEED = 42


def validate_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA nao esta disponivel. Este experimento YOLO foi configurado para device=0.")
    return torch.cuda.get_device_name(0)


def list_test_images(data_yaml_path: Path) -> list[Path]:
    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    dataset_root = Path(data.get("path", data_yaml_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml_path.parent / dataset_root
    test_dir = Path(data.get("test", "test/images"))
    if not test_dir.is_absolute():
        test_dir = dataset_root / test_dir
    return sorted(path for path in test_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["yolo"]
    data_yaml = ROOT_DIR / config["data_yaml"]
    model_path = BEST_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo YOLO nao encontrado: {model_path}. "
            "Execute primeiro src/training/train_yolo.py."
        )
    images = list_test_images(data_yaml)
    if not images:
        raise FileNotFoundError("Nenhuma imagem encontrada no split test do YOLO.")

    selected_images = images[:]
    random.Random(RANDOM_SEED).shuffle(selected_images)
    selected_images = selected_images[: min(MAX_IMAGES, len(selected_images))]

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    gpu_name = validate_cuda()
    print(f"Gerando predicoes visuais para {len(selected_images)} imagens")
    print(f"GPU: {gpu_name}")
    model.predict(
        source=[str(path) for path in selected_images],
        conf=0.25,
        device=int(config["device"]),
        save=True,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
    )

    print(f"Predicoes salvas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
