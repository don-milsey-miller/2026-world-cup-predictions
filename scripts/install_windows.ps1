param(
    [switch]$Dev,
    [switch]$DownloadData,
    [switch]$TrainModel,
    [switch]$RunChecks
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

function Find-Python {
    if (Test-Python -Exe "python" -Args @()) {
        return @{ Exe = "python"; Args = @() }
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        if (Test-Python -Exe "py" -Args @("-3.11")) {
            return @{ Exe = "py"; Args = @("-3.11") }
        }
        if (Test-Python -Exe "py" -Args @("-3")) {
            return @{ Exe = "py"; Args = @("-3") }
        }
    }

    throw "Python 3.11+ was not found. Install it from https://www.python.org/downloads/windows/."
}

function Test-Python {
    param(
        [string]$Exe,
        [string[]]$Args
    )

    if (-not (Get-Command $Exe -ErrorAction SilentlyContinue)) {
        return $false
    }

    & $Exe @Args -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
    return $LASTEXITCODE -eq 0
}

function Invoke-BasePython {
    & $script:PythonCommand.Exe @($script:PythonCommand.Args + $args)
}

Push-Location $ProjectRoot
try {
    $script:PythonCommand = Find-Python

    if (-not (Test-Path $VenvPython)) {
        Write-Host "Creating virtual environment at $VenvPath"
        Invoke-BasePython -m venv $VenvPath
    }

    Write-Host "Upgrading pip"
    & $VenvPython -m pip install --upgrade pip

    $InstallTarget = "."
    if ($Dev) {
        $InstallTarget = ".[dev]"
    }

    Write-Host "Installing project: $InstallTarget"
    & $VenvPython -m pip install -e $InstallTarget

    if ($DownloadData) {
        Write-Host "Downloading latest source data"
        & $VenvPython scripts/download_data.py
    }

    if ($TrainModel) {
        Write-Host "Training model artifact"
        & $VenvPython scripts/train_model.py
    }

    if ($RunChecks) {
        if (-not $Dev) {
            Write-Host "Installing dev tools for checks"
            & $VenvPython -m pip install -e ".[dev]"
        }
        & $VenvPython -m ruff format --check .
        & $VenvPython -m ruff check .
        & $VenvPython -m pytest
    }

    Write-Host ""
    Write-Host "Install complete."
    Write-Host "Run the app with:"
    Write-Host "  .\.venv\Scripts\python -m uvicorn worldcup_predictor.app:app --reload"
    Write-Host "Then open http://127.0.0.1:8000"
} finally {
    Pop-Location
}
