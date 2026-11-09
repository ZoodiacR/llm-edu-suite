# Windows (PowerShell): create the project virtual environment and register the Jupyter kernel.
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1
$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv")) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
# PyTorch: the CUDA build if an NVIDIA GPU is present (much faster training), otherwise the small CPU build.
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "NVIDIA GPU detected -> installing PyTorch with CUDA"
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
} else {
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}
& $py -m pip install -r requirements-dev.txt
& $py -m ipykernel install --user --name llm_edu_suite --display-name "Python (llm_edu_suite)"
& $py -m pytest -q
& $py -c "import torch; print('PyTorch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')"
Write-Host "`nListo. Activa el entorno con:  .\.venv\Scripts\Activate.ps1   y abre:  jupyter lab notebooks/"
