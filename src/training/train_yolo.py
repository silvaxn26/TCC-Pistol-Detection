from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "training_config.yaml"
SUMMARY_PATH = ROOT_DIR / "results" / "reports" / "yolo_training_summary.md"


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def validate_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA nao esta disponivel neste ambiente. "
            "Verifique PyTorch com CUDA antes de treinar com device=0."
        )
    return torch.cuda.get_device_name(0)


def validate_plot_dependencies() -> None:
    try:
        import matplotlib  # noqa: F401
        import kiwisolver  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Dependencias de plotagem do Ultralytics/Matplotlib estao quebradas. "
            "Execute na .venv ativa: "
            "pip install --force-reinstall --no-cache-dir --no-deps "
            "matplotlib==3.8.4 contourpy==1.2.1 kiwisolver==1.4.5"
        ) from error


def copy_final_weights(run_dir: Path, weights_dir: Path) -> tuple[Path, Path]:
    source_best = run_dir / "weights" / "best.pt"
    source_last = run_dir / "weights" / "last.pt"
    if not source_best.exists():
        raise FileNotFoundError(f"best.pt nao encontrado apos o treinamento: {source_best}")
    if not source_last.exists():
        raise FileNotFoundError(f"last.pt nao encontrado apos o treinamento: {source_last}")

    weights_dir.mkdir(parents=True, exist_ok=True)
    target_best = weights_dir / "best.pt"
    target_last = weights_dir / "last.pt"
    shutil.copy2(source_best, target_best)
    shutil.copy2(source_last, target_last)
    return target_best, target_last


def read_last_training_row(results_csv: Path) -> dict[str, float | str]:
    if not results_csv.exists():
        return {}
    try:
        import pandas as pd
    except ImportError as error:
        print(
            "Aviso: nao foi possivel importar pandas para ler results.csv. "
            "O treinamento nao sera interrompido; o resumo sera gerado sem a ultima linha de metricas. "
            "Para corrigir o ambiente, execute a ordem de instalacao documentada no README. "
            f"Erro original: {error}"
        )
        return {}

    df = pd.read_csv(results_csv)
    if df.empty:
        return {}

    last_row = {}
    for key, value in df.iloc[-1].to_dict().items():
        if hasattr(value, "item"):
            value = value.item()
        last_row[str(key).strip()] = value if isinstance(value, (int, float, str)) else str(value)
    return last_row


def write_summary(
    config: dict,
    gpu_name: str,
    run_dir: Path,
    best_path: Path,
    last_path: Path,
    final_training_row: dict[str, float | str],
) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Resumo do treinamento YOLOv8 Nano",
        "",
        f"- Data/hora: {datetime.now().isoformat(timespec='seconds')}",
        f"- GPU: `{gpu_name}`",
        f"- Modelo base: `{config['model']}`",
        f"- Dataset: `{config['data_yaml']}`",
        f"- Epochs: `{config['epochs']}`",
        f"- Imagem: `{config['imgsz']}`",
        f"- Batch: `{config['batch']}`",
        f"- Patience: `{config['patience']}`",
        f"- Device: `{config['device']}`",
        f"- Workers: `{config['workers']}`",
        f"- Experimento Ultralytics: `{relative(run_dir)}`",
        f"- best.pt final: `{relative(best_path)}`",
        f"- last.pt final: `{relative(last_path)}`",
        "",
        "## Observacao",
        "",
        "Este resumo registra apenas artefatos gerados pela execucao real do treinamento. Nao edite metricas manualmente.",
        "",
    ]

    if final_training_row:
        lines.extend(["## Ultima linha de metricas do treino", "", "```json"])
        lines.append(json.dumps(final_training_row, indent=2, ensure_ascii=False))
        lines.extend(["```", ""])

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["yolo"]
    data_yaml = ROOT_DIR / config["data_yaml"]
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YOLO processado nao encontrado: {data_yaml}. "
            "Execute primeiro src/dataset/check_processed_dataset.py."
        )

    gpu_name = validate_cuda()
    validate_plot_dependencies()
    print("CUDA disponivel para treinamento YOLOv8 Nano")
    print(f"GPU: {gpu_name}")

    project_dir = ROOT_DIR / config["project"]
    run_dir = project_dir / config["name"]
    weights_dir = ROOT_DIR / config["weights_dir"]

    print("Iniciando treinamento YOLOv8 Nano")
    print(f"Dataset: {data_yaml}")
    print(f"Saida do treinamento: {run_dir}")
    print(f"Pesos finais serao copiados para: {weights_dir}")

    model = YOLO(config["model"])
    model.train(
        data=str(data_yaml),
        epochs=int(config["epochs"]),
        imgsz=int(config["imgsz"]),
        batch=int(config["batch"]),
        patience=int(config["patience"]),
        device=int(config["device"]),
        workers=int(config["workers"]),
        project=str(project_dir),
        name=config["name"],
        plots=True,
        exist_ok=True,
    )

    best_path, last_path = copy_final_weights(run_dir, weights_dir)
    final_training_row = read_last_training_row(run_dir / "results.csv")
    write_summary(config, gpu_name, run_dir, best_path, last_path, final_training_row)

    print("Treinamento concluido.")
    print(f"best.pt: {best_path}")
    print(f"last.pt: {last_path}")
    print(f"Resumo salvo em: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
