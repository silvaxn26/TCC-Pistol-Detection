from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "training_config.yaml"
SUMMARY_PATH = ROOT_DIR / "results" / "reports" / "faster_rcnn_training_summary.md"
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.25


def print_torch_diagnostics() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA compilado no PyTorch: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")


def get_training_device(config: dict) -> torch.device:
    requested_device = str(config.get("device", "cuda")).lower()
    require_cuda = bool(config.get("require_cuda", False))

    print_torch_diagnostics()

    if requested_device in {"cuda", "0", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        message = (
            "CUDA nao esta disponivel nesta virtualenv. "
            "Reinstale torch/torchvision/torchaudio com suporte CUDA 11.8:\n"
            "pip uninstall torch torchvision torchaudio -y\n"
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
            "python -c \"import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())\""
        )
        if require_cuda:
            raise RuntimeError(message)
        print(f"Aviso: {message}")
        return torch.device("cpu")

    return torch.device("cpu")


def find_image(split_dir: Path, file_name: str) -> Path:
    candidates = [split_dir / file_name, split_dir / "images" / file_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Imagem COCO nao encontrada: {file_name} em {split_dir}")


def safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
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


class CocoPistolDataset(Dataset):
    def __init__(self, split_dir: Path):
        self.split_dir = split_dir
        self.annotation_file = split_dir / "_annotations.coco.json"
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Anotacao COCO nao encontrada: {self.annotation_file}")
        data = json.loads(self.annotation_file.read_text(encoding="utf-8"))
        self.images = []
        for image in data.get("images", []):
            normalized_image = dict(image)
            normalized_image["id"] = safe_int(image.get("id"))
            self.images.append(normalized_image)
        self.images = sorted(self.images, key=lambda item: item["id"])
        self.annotations_by_image_id: dict[int, list[dict]] = {image["id"]: [] for image in self.images}
        for annotation in data.get("annotations", []):
            image_id = safe_int(annotation.get("image_id"))
            self.annotations_by_image_id.setdefault(image_id, []).append(annotation)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_info = self.images[index]
        image_id = image_info["id"]
        image_path = find_image(self.split_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for annotation in self.annotations_by_image_id.get(image_id, []):
            parsed_bbox = parse_coco_bbox(annotation)
            if parsed_bbox is None:
                continue
            x, y, width, height = parsed_bbox
            boxes.append([x, y, x + width, y + height])
            labels.append(1)
            areas.append(safe_float(annotation.get("area"), width * height))
            iscrowd.append(safe_int(annotation.get("iscrowd"), 0))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor([image_id], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return F.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int):
    try:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        print("Usando pesos pre-treinados DEFAULT do torchvision.")
    except Exception as error:
        print(f"Aviso: nao foi possivel carregar pesos pre-treinados. Treinando sem pesos iniciais. Erro: {error}")
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def move_targets_to_device(targets, device: torch.device):
    return [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]


def train_one_epoch(model, loader, optimizer, device: torch.device, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch:03d} treino", unit="batch")
    for images, targets in progress:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets_to_device(targets, device)
        losses = sum(loss for loss in model(images, targets).values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += float(losses.item())
        progress.set_postfix(loss=f"{float(losses.item()):.4f}")
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def compute_validation_loss(model, loader, device: torch.device, epoch: int) -> float:
    # Torchvision detection models return losses only when targets are passed in train mode.
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch:03d} valid_loss", unit="batch")
    for images, targets in progress:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets_to_device(targets, device)
        losses = sum(loss for loss in model(images, targets).values())
        total_loss += float(losses.item())
        progress.set_postfix(loss=f"{float(losses.item()):.4f}")
    model.eval()
    return total_loss / max(len(loader), 1)


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


def match_predictions(predicted_boxes, predicted_scores, gt_boxes) -> tuple[int, int, int]:
    keep = predicted_scores >= CONFIDENCE_THRESHOLD
    predictions = predicted_boxes[keep].detach().cpu().tolist()
    scores = predicted_scores[keep].detach().cpu().tolist()
    ordered = [box for _, box in sorted(zip(scores, predictions), reverse=True)]
    gt_list = gt_boxes.detach().cpu().tolist()
    matched_gt = set()
    tp = 0
    fp = 0

    for predicted_box in ordered:
        best_iou = 0.0
        best_index = None
        for index, gt_box in enumerate(gt_list):
            if index in matched_gt:
                continue
            current_iou = box_iou(predicted_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = index
        if best_index is not None and best_iou >= IOU_THRESHOLD:
            matched_gt.add(best_index)
            tp += 1
        else:
            fp += 1
    fn = len(gt_list) - tp
    return tp, fp, fn


@torch.no_grad()
def compute_validation_iou_metrics(model, loader, device: torch.device, epoch: int) -> dict[str, float | int]:
    model.eval()
    tp = 0
    fp = 0
    fn = 0
    progress = tqdm(loader, desc=f"Epoch {epoch:03d} metricas IoU", unit="batch")
    for images, targets in progress:
        images = [image.to(device, non_blocking=True) for image in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            image_tp, image_fp, image_fn = match_predictions(
                output["boxes"],
                output["scores"],
                target["boxes"],
            )
            tp += image_tp
            fp += image_fp
            fn += image_fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"valid_precision": precision, "valid_recall": recall, "valid_f1": f1, "valid_tp": tp, "valid_fp": fp, "valid_fn": fn}


def save_checkpoint(path: Path, model, optimizer, epoch: int, train_loss: float, valid_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "num_classes": 2,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0))


def write_logs_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "valid_loss", "valid_precision", "valid_recall", "valid_f1", "valid_tp", "valid_fp", "valid_fn"],
        )
        writer.writeheader()
        writer.writerows(rows)


def read_logs_csv(path: Path) -> list[dict[str, float | int]]:
    if not path.exists():
        return []
    rows: list[dict[str, float | int]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "valid_loss": float(row["valid_loss"]),
                    "valid_precision": float(row["valid_precision"]),
                    "valid_recall": float(row["valid_recall"]),
                    "valid_f1": float(row["valid_f1"]),
                    "valid_tp": int(float(row["valid_tp"])),
                    "valid_fp": int(float(row["valid_fp"])),
                    "valid_fn": int(float(row["valid_fn"])),
                }
            )
    return rows


def write_summary(config: dict, device: torch.device, best_valid_loss: float, rows: list[dict[str, float | int]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Resumo do treinamento Faster R-CNN",
        "",
        f"- Data/hora: {datetime.now().isoformat(timespec='seconds')}",
        f"- Modelo: `fasterrcnn_resnet50_fpn`",
        f"- Pesos pre-treinados: `FasterRCNN_ResNet50_FPN_Weights.DEFAULT`, quando disponivel",
        f"- Dataset COCO: `{config['coco_dir']}`",
        f"- Epochs: `{config['epochs']}`",
        f"- Batch size: `{config['batch_size']}`",
        f"- Learning rate: `{config['learning_rate']}`",
        f"- Momentum: `{config['momentum']}`",
        f"- Weight decay: `{config['weight_decay']}`",
        f"- Num workers: `{config['num_workers']}`",
        f"- Device: `{device}`",
        f"- IoU de validacao: `{IOU_THRESHOLD}`",
        f"- Melhor valid_loss: `{best_valid_loss:.6f}`",
        f"- best.pt: `{config['output_dir']}/best.pt`",
        f"- last.pt: `{config['output_dir']}/last.pt`",
        "",
        "## Historico de perdas",
        "",
        "| Epoch | Train loss | Valid loss | Valid precision | Valid recall | Valid F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        precision = row["valid_precision"]
        recall = row["valid_recall"]
        f1 = row["valid_f1"]
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.6f} | {row['valid_loss']:.6f} | "
            f"{precision:.6f} | {recall:.6f} | {f1:.6f} |"
        )
    lines.append("")
    lines.append("As metricas finais de deteccao devem ser geradas por `src/evaluation/evaluate_faster_rcnn.py`.")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["faster_rcnn"]
    coco_dir = ROOT_DIR / config["coco_dir"]
    device = get_training_device(config)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_dataset = CocoPistolDataset(coco_dir / "train")
    valid_dataset = CocoPistolDataset(coco_dir / "valid")
    num_workers = int(config["num_workers"])
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    print(f"Treinando Faster R-CNN em: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(num_classes=int(config["num_classes"])).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
    )

    output_dir = ROOT_DIR / config["output_dir"]
    logs_dir = ROOT_DIR / config["logs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    history = read_logs_csv(logs_dir / "training_log.csv")
    best_valid_loss = min((float(row["valid_loss"]) for row in history), default=float("inf"))
    start_epoch = 1
    last_checkpoint = output_dir / "last.pt"
    if bool(config.get("resume", True)) and last_checkpoint.exists():
        completed_epoch = load_checkpoint(last_checkpoint, model, optimizer, device)
        start_epoch = completed_epoch + 1
        print(f"Checkpoint encontrado: {last_checkpoint}")
        print(f"Retomando treinamento a partir da epoch {start_epoch:03d}")

    total_epochs = int(config["epochs"])
    if start_epoch > total_epochs:
        print(f"Treinamento ja possui {total_epochs} epochs concluida(s). Nada a retomar.")
        write_summary(config, device, best_valid_loss, history)
        return

    validation_metrics_every = int(config.get("validation_metrics_every", 1))
    last_valid_metrics = {"valid_precision": 0.0, "valid_recall": 0.0, "valid_f1": 0.0, "valid_tp": 0, "valid_fp": 0, "valid_fn": 0}
    if history:
        last_row = history[-1]
        last_valid_metrics = {
            "valid_precision": float(last_row["valid_precision"]),
            "valid_recall": float(last_row["valid_recall"]),
            "valid_f1": float(last_row["valid_f1"]),
            "valid_tp": int(last_row["valid_tp"]),
            "valid_fp": int(last_row["valid_fp"]),
            "valid_fn": int(last_row["valid_fn"]),
        }

    for epoch in range(start_epoch, total_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        valid_loss = compute_validation_loss(model, valid_loader, device, epoch)
        should_compute_iou = epoch == 1 or epoch % validation_metrics_every == 0 or epoch == total_epochs
        if should_compute_iou:
            valid_metrics = compute_validation_iou_metrics(model, valid_loader, device, epoch)
            last_valid_metrics = valid_metrics
        else:
            valid_metrics = last_valid_metrics
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, **valid_metrics})
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
            f"valid_f1={valid_metrics['valid_f1']:.6f}"
        )

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, train_loss, valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, train_loss, valid_loss)
            print(f"Novo melhor modelo salvo em: {output_dir / 'best.pt'}")

        write_logs_csv(logs_dir / "training_log.csv", history)

    shutil.copy2(output_dir / "best.pt", logs_dir / "best.pt")
    shutil.copy2(output_dir / "last.pt", logs_dir / "last.pt")
    write_summary(config, device, best_valid_loss, history)
    print(f"Treinamento Faster R-CNN concluido. Checkpoints em: {output_dir}")
    print(f"Relatorio salvo em: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
