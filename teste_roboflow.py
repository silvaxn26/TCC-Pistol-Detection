from roboflow import Roboflow
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise RuntimeError("ROBOFLOW_API_KEY não encontrada no .env")

rf = Roboflow(api_key=api_key)

workspace = "silvaxn"
project_slug = "pistol_detection"
version_number = 1

print("Carregando workspace...")
workspace_obj = rf.workspace(workspace)

print("Carregando projeto...")
project = workspace_obj.project(project_slug)

print("Baixando YOLOv8...")
dataset_yolo = project.version(version_number).download("yolov8")
print("YOLO location:", dataset_yolo.location)

print("\nArquivos YOLO encontrados:")
for path in Path(dataset_yolo.location).rglob("*"):
    print(path)

print("\nBaixando COCO...")
dataset_coco = project.version(version_number).download("coco")
print("COCO location:", dataset_coco.location)

print("\nArquivos COCO encontrados:")
for path in Path(dataset_coco.location).rglob("*"):
    print(path)