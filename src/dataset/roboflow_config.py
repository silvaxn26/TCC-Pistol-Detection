from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from roboflow.adapters.rfapi import RoboflowError


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RoboflowDatasetConfig:
    api_key: str
    workspace: str
    project: str
    version: int


def load_roboflow_config() -> RoboflowDatasetConfig:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("Defina ROBOFLOW_API_KEY no arquivo .env antes de executar o download.")

    return RoboflowDatasetConfig(
        api_key=api_key,
        workspace=os.getenv("ROBOFLOW_WORKSPACE", "silvaxn"),
        project=os.getenv("ROBOFLOW_PROJECT", "pistol_detection"),
        version=int(os.getenv("ROBOFLOW_VERSION", "1")),
    )


def explain_project_error(error: RoboflowError, config: RoboflowDatasetConfig) -> RoboflowError:
    message = (
        f"{error}\n\n"
        "Nao foi possivel carregar o projeto no Roboflow.\n"
        f"Configuracao usada: workspace={config.workspace}, project={config.project}, version={config.version}.\n"
        "Confira se ROBOFLOW_PROJECT no .env usa o slug do projeto, nao apenas o nome visual. "
        "Exemplos comuns: pistol_detection, pistol-detection ou o slug exibido na URL do Roboflow Universe.\n"
        "Confira tambem se a API Key tem permissao para acessar esse workspace/projeto."
    )
    return RoboflowError(message)

