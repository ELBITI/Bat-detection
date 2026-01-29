# Script d'installation et lancement - YOLOv8 Streamlit App
# Utilisez Python 3.10, 3.11 ou 3.12 (Python 3.13 n'est pas encore supporte par ultralytics)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== YOLOv8 Streamlit - Installation et lancement ===" -ForegroundColor Cyan
Write-Host ""

$pyVer = python --version 2>&1
Write-Host "Python detecte: $pyVer"

# Verifier version Python (3.13 non supporte par ultralytics)
$version = (python -c "import sys; print(sys.version_info.minor)" 2>$null)
if ($version -eq 13) {
    Write-Host ""
    Write-Host "ATTENTION: Python 3.13 n'est pas encore supporte par 'ultralytics'." -ForegroundColor Yellow
    Write-Host "Installez Python 3.10, 3.11 ou 3.12 depuis https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Puis creez un venv et relancez ce script." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Appuyez sur Entree pour quitter"
    exit 1
}

# Installation des dependances
Write-Host ""
Write-Host "Installation des bibliotheques..." -ForegroundColor Green
pip install -r requirements-minimal.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Erreur lors de l'installation. Essayez: pip install streamlit ultralytics opencv-python-headless easyocr Pillow pytube numpy" -ForegroundColor Red
    Read-Host "Appuyez sur Entree pour quitter"
    exit 1
}

# Lancer l'app
Write-Host ""
Write-Host "Lancement de l'interface Streamlit..." -ForegroundColor Green
Write-Host "Ouvrez votre navigateur sur l'URL affichee (souvent http://localhost:8501)" -ForegroundColor Cyan
Write-Host "Pour arreter: Ctrl+C dans cette fenetre" -ForegroundColor Gray
Write-Host ""
streamlit run app.py
