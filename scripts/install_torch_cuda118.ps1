Write-Host "Instalacao do PyTorch CUDA 11.8 para o projeto do TCC"
Write-Host ""

$pythonVersion = python --version
$pipVersion = python -m pip --version

Write-Host "Python em uso: $pythonVersion"
Write-Host "pip em uso: $pipVersion"
Write-Host ""

if ($pythonVersion -notmatch "Python 3\.11") {
    Write-Host "ATENCAO: este projeto foi validado com Python 3.11."
    Write-Host "Crie/ative uma virtualenv com Python 3.11 antes de instalar o PyTorch CUDA."
    Write-Host "Exemplo:"
    Write-Host "py -3.11 -m venv .venv"
    Write-Host ".venv\Scripts\activate"
    exit 1
}

python -m pip install --upgrade pip
python -m pip uninstall torch torchvision torchaudio -y

python -m pip install `
    torch==2.7.1+cu118 `
    torchvision==0.22.1+cu118 `
    torchaudio==2.7.1+cu118 `
    --index-url https://download.pytorch.org/whl/cu118

Write-Host ""
Write-Host "Verificando CUDA:"
python src/utils/check_cuda.py
