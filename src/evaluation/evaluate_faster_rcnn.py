from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import torch
import yaml
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "training_config.yaml"
MODEL_PATH = ROOT_DIR / "models" / "faster_rcnn" / "best.pt"
METRICS_DIR = ROOT_DIR / "results" / "metrics"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
MAP_THRESHOLDS = [round(0.5 + index * 0.05, 2) for index in range(10)]


def build_model(num_classes: int = 2):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(device: torch.device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo Faster R-CNN nao encontrado: {MODEL_PATH}")
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


def safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_coco_bbox(annotation: dict) -> tuple[float, float, float, float] | None:
    """Converte bbox COCO para float e ignora caixas invalidas."""
    try:
        x, y, width, height = [float(value) for value in annotation.get("bbox", [])]
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def load_coco_split(split_dir: Path) -> tuple[list[dict], dict[int, list[tuple[float, float, float, float]]]]:
    annotation_file = split_dir / "_annotations.coco.json"
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = []
    for image in data["images"]:
        normalized_image = dict(image)
        normalized_image["id"] = safe_int(image.get("id"))
        images.append(normalized_image)
    gt_by_image: dict[int, list[tuple[float, float, float, float]]] = {image["id"]: [] for image in images}
    for annotation in data["annotations"]:
        parsed_bbox = parse_coco_bbox(annotation)
        if parsed_bbox is None:
            continue
        x, y, width, height = parsed_bbox
        image_id = safe_int(annotation.get("image_id"))
        gt_by_image.setdefault(image_id, []).append((x, y, x + width, y + height))
    return images, gt_by_image


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


def average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def compute_ap(predictions: list[dict], gt_by_image: dict[int, list], iou_threshold: float) -> float:
    sorted_predictions = sorted(predictions, key=lambda item: item["score"], reverse=True)
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    matched = {image_id: set() for image_id in gt_by_image}
    tp = []
    fp = []

    for prediction in sorted_predictions:
        image_id = prediction["image_id"]
        best_iou = 0.0
        best_index = None
        for index, gt_box in enumerate(gt_by_image.get(image_id, [])):
            if index in matched[image_id]:
                continue
            current_iou = box_iou(prediction["box"], gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = index
        if best_index is not None and best_iou >= iou_threshold:
            matched[image_id].add(best_index)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    cumulative_tp = 0
    cumulative_fp = 0
    recalls = []
    precisions = []
    for tp_value, fp_value in zip(tp, fp):
        cumulative_tp += tp_value
        cumulative_fp += fp_value
        recalls.append(cumulative_tp / total_gt if total_gt else 0.0)
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp) if cumulative_tp + cumulative_fp else 0.0)
    return average_precision(recalls, precisions)


def compute_confusion_counts(predictions: list[dict], gt_by_image: dict[int, list], iou_threshold: float):
    filtered_predictions = [prediction for prediction in predictions if prediction["score"] >= CONFIDENCE_THRESHOLD]
    matched = {image_id: set() for image_id in gt_by_image}
    tp = 0
    fp = 0
    for prediction in sorted(filtered_predictions, key=lambda item: item["score"], reverse=True):
        image_id = prediction["image_id"]
        best_iou = 0.0
        best_index = None
        for index, gt_box in enumerate(gt_by_image.get(image_id, [])):
            if index in matched[image_id]:
                continue
            current_iou = box_iou(prediction["box"], gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = index
        if best_index is not None and best_iou >= iou_threshold:
            matched[image_id].add(best_index)
            tp += 1
        else:
            fp += 1
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    fn = total_gt - tp
    return tp, fp, fn


@torch.no_grad()
def collect_predictions(model, split_dir: Path, images: list[dict], device: torch.device):
    predictions = []
    inference_times = []
    for image_info in images:
        image_path = find_image(split_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        tensor = F.to_tensor(image).to(device)
        start = perf_counter()
        output = model([tensor])[0]
        inference_times.append(perf_counter() - start)
        for box, score, label in zip(output["boxes"].detach().cpu(), output["scores"].detach().cpu(), output["labels"].detach().cpu()):
            if int(label) != 1:
                continue
            predictions.append(
                {
                    "image_id": image_info["id"],
                    "score": float(score),
                    "box": tuple(float(value) for value in box.tolist()),
                }
            )
    mean_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    return predictions, mean_time


def save_csv(path: Path, metrics: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["faster_rcnn"]
    split_dir = ROOT_DIR / config["coco_dir"] / "test"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Avaliando Faster R-CNN em: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = load_model(device)
    images, gt_by_image = load_coco_split(split_dir)
    predictions, mean_inference_seconds = collect_predictions(model, split_dir, images, device)

    tp, fp, fn = compute_confusion_counts(predictions, gt_by_image, IOU_THRESHOLD)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    map50 = compute_ap(predictions, gt_by_image, 0.5)
    map50_95 = sum(compute_ap(predictions, gt_by_image, threshold) for threshold in MAP_THRESHOLDS) / len(MAP_THRESHOLDS)

    metrics = {
        "model": str(MODEL_PATH.relative_to(ROOT_DIR)),
        "data": str(split_dir.relative_to(ROOT_DIR)),
        "split": "test",
        "test_images": len(images),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "map50": map50,
        "map50_95": map50_95,
        "mean_inference_seconds": mean_inference_seconds,
        "false_positives": fp,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = METRICS_DIR / "faster_rcnn_metrics.json"
    csv_path = METRICS_DIR / "faster_rcnn_metrics.csv"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_csv(csv_path, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Metricas salvas em: {json_path}")


if __name__ == "__main__":
    main()
