from __future__ import annotations

from pathlib import Path
from time import perf_counter

import streamlit as st
from PIL import Image, ImageDraw


ROOT_DIR = Path(__file__).resolve().parents[1]


st.set_page_config(page_title="TCC Pistol Detection", layout="wide")
st.title("Comparacao YOLOv8 Nano x Faster R-CNN")

uploaded_file = st.file_uploader("Enviar imagem", type=["jpg", "jpeg", "png", "bmp", "webp"])

yolo_model_path = st.text_input(
    "Modelo YOLOv8",
    str(ROOT_DIR / "models" / "yolo" / "yolov8n_pistol" / "weights" / "best.pt"),
)
faster_model_path = st.text_input(
    "Modelo Faster R-CNN",
    str(ROOT_DIR / "models" / "faster_rcnn" / "faster_rcnn_pistol.pth"),
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_container_width=True)
    col_yolo, col_faster = st.columns(2)

    with col_yolo:
        st.subheader("YOLOv8 Nano")
        if st.button("Detectar com YOLOv8"):
            if not Path(yolo_model_path).exists():
                st.error("Modelo YOLOv8 nao encontrado. Treine o modelo antes da demonstracao.")
            else:
                from ultralytics import YOLO

                model = YOLO(yolo_model_path)
                start = perf_counter()
                results = model.predict(image, conf=0.25)
                elapsed = perf_counter() - start
                st.write(f"Tempo de inferencia: {elapsed:.4f} s")
                st.image(results[0].plot(), channels="BGR", use_container_width=True)

    with col_faster:
        st.subheader("Faster R-CNN")
        if st.button("Detectar com Faster R-CNN"):
            if not Path(faster_model_path).exists():
                st.error("Modelo Faster R-CNN nao encontrado. Treine o modelo antes da demonstracao.")
            else:
                import torch
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
                from torchvision.transforms import functional as F

                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                model = fasterrcnn_resnet50_fpn(weights=None)
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
                model.load_state_dict(torch.load(faster_model_path, map_location=device))
                model.to(device)
                model.eval()

                tensor = F.to_tensor(image).to(device)
                start = perf_counter()
                with torch.no_grad():
                    prediction = model([tensor])[0]
                elapsed = perf_counter() - start

                output_image = image.copy()
                draw = ImageDraw.Draw(output_image)
                keep = prediction["scores"].detach().cpu() >= 0.5
                for box, score in zip(prediction["boxes"].detach().cpu()[keep], prediction["scores"].detach().cpu()[keep]):
                    x1, y1, x2, y2 = [float(value) for value in box]
                    draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
                    draw.text((x1, max(0, y1 - 18)), f"pistol {float(score):.2f}", fill="lime")

                st.write(f"Tempo de inferencia: {elapsed:.4f} s")
                st.image(output_image, use_container_width=True)

st.divider()
st.caption(
    "Interface inicial para demonstracao academica. As metricas so devem ser exibidas apos execucoes reais."
)
