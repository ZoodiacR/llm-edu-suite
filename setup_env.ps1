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
& $py -m pip install -r requirements.txt
& $py -m ipykernel install --user --name llm-edu-suite --display-name "Python (llm-edu-suite)"

# Local LLMs for notebooks 10-16 (Ollama). Install it from https://ollama.com if it is missing.
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    ollama pull qwen2.5:3b
    ollama pull nomic-embed-text
} else {
    Write-Host "Ollama not found: install it from https://ollama.com, then run: ollama pull qwen2.5:3b; ollama pull nomic-embed-text"
}
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host "Created .env from .env.example (add your LANGSMITH_API_KEY for tracing)" }
& $py -m pytest -q -m "not ollama"
& $py -c "import torch; print('PyTorch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')"
Write-Host "`nListo. Activa el entorno con:  .\.venv\Scripts\Activate.ps1   y abre:  jupyter lab notebooks/"
