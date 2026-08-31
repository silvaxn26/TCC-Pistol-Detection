from __future__ import annotations

import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA compilado no PyTorch: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total de GPUs: {torch.cuda.device_count()}")
    else:
        print("")
        print("CUDA nao esta disponivel nesta virtualenv.")
        print("Reinstale o PyTorch com suporte CUDA 11.8:")
        print("pip uninstall torch torchvision torchaudio -y")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")


if __name__ == "__main__":
    main()
