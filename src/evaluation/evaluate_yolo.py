from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import torch
import yaml
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "training_config.yaml"
METRICS_DIR = ROOT_DIR / "results" / "metrics"
BEST_MODEL_PATH = ROOT_DIR / "models" / "yolo" / "best.pt"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA nao esta disponivel. Este experimento YOLO foi configurado para device=0.")
    return torch.cuda.get_device_name(0)


def list_test_images(data_yaml_path: Path) -> list[Path]:
    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    dataset_root = Path(data.get("path", data_yaml_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml_path.parent / dataset_root
    test_value = data.get("test", "test/images")
    test_dir = Path(test_value)
    if not test_dir.is_absolute():
        test_dir = dataset_root / test_dir
    return sorted(path for path in test_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_path_for_image(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def yolo_box_to_xyxy(values: list[float], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x_center, y_center, width, height = values
    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height
    return x1, y1, x2, y2


def read_ground_truth_boxes(image_path: Path, image_width: int, image_height: int) -> list[tuple[float, float, float, float]]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        if class_id != 0:
            continue
        boxes.append(yolo_box_to_xyxy([float(value) for value in parts[1:]], image_width, image_height))
    return boxes


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def count_false_positives(model: YOLO, images: list[Path], device: int, confidence: float = 0.25, iou_threshold: float = 0.5) -> tuple[int, float]:
    false_positives = 0
    inference_times = []
    for image_path in images:
        start = perf_counter()
        results = model.predict(str(image_path), conf=confidence, device=device, verbose=False)
        inference_times.append(perf_counter() - start)
        if not results:
            continue
        result = results[0]
        image_height, image_width = result.orig_shape
        ground_truth_boxes = read_ground_truth_boxes(image_path, image_width, image_height)
        matched_ground_truth = set()
        # Pareia predicoes com ground truth por IoU para estimar falsos positivos.
        for predicted_box in result.boxes.xyxy.cpu().tolist():
            best_iou = 0.0
            best_index = None
            predicted_tuple = tuple(float(value) for value in predicted_box)
            for index, ground_truth_box in enumerate(ground_truth_boxes):
                if index in matched_ground_truth:
                    continue
                current_iou = box_iou(predicted_tuple, ground_truth_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_index = index
            if best_index is not None and best_iou >= iou_threshold:
                matched_ground_truth.add(best_index)
            else:
                false_positives += 1
    mean_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    return false_positives, mean_time


def save_csv(path: Path, metrics: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["yolo"]
    data_yaml = ROOT_DIR / config["data_yaml"]
    model_path = BEST_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo YOLO nao encontrado: {model_path}. "
            "Execute primeiro src/training/train_yolo.py."
        )
    model = YOLO(str(model_path))
    device = int(config["device"])
    gpu_name = validate_cuda()

    print(f"Avaliando YOLOv8 Nano com modelo: {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"GPU: {gpu_name}")

    validation = model.val(
        data=str(data_yaml),
        split="test",
        device=device,
        project=str(ROOT_DIR / "results" / "evaluation"),
        name="yolo_test",
        plots=True,
        exist_ok=True,
    )

    test_images = list_test_images(data_yaml)
    false_positives, mean_inference_seconds = count_false_positives(model, test_images, device=device)

    precision = float(validation.box.mp)
    recall = float(validation.box.mr)
    f1_denominator = precision + recall
    metrics = {
        "model": str(model_path.relative_to(ROOT_DIR)),
        "data": str(data_yaml.relative_to(ROOT_DIR)),
        "device": device,
        "gpu": gpu_name,
        "split": "test",
        "test_images": len(test_images),
        "precision": precision,
        "recall": recall,
        "f1_score": 2 * precision * recall / f1_denominator if f1_denominator else 0.0,
        "map50": float(validation.box.map50),
        "map50_95": float(validation.box.map),
        "mean_inference_seconds": mean_inference_seconds,
        "false_positives": false_positives,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = METRICS_DIR / "yolo_metrics.json"
    csv_path = METRICS_DIR / "yolo_metrics.csv"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_csv(csv_path, metrics)

    print(json.dumps(metrics, indent=2))
    print(f"Metricas salvas em: {json_path}")
    print(f"Metricas CSV salvas em: {csv_path}")


if __name__ == "__main__":
    main()
