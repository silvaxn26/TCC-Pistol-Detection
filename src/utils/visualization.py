from __future__ import annotations

from pathlib import Path

import cv2


def draw_boxes(
    image_path: Path,
    boxes: list[tuple[float, float, float, float]],
    labels: list[str],
    scores: list[float] | None = None,
) -> object:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Imagem nao encontrada ou invalida: {image_path}")

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(value) for value in box]
        label = labels[index] if index < len(labels) else "object"
        if scores and index < len(scores):
            label = f"{label} {scores[index]:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    return image


def save_image(path: Path, image: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

