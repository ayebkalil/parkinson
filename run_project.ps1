param(
    [switch]$Install,
    [switch]$Build,
    [switch]$NoBrowser,
    [ValidateRange(1024, 65535)][int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'La création du venv a échoué.' }
        $Install = $true
    }
    # Use the virtual environment directly; no PowerShell execution-policy change is needed.
    $env:VIRTUAL_ENV = Join-Path $projectRoot '.venv'
    $env:Path = (Join-Path $env:VIRTUAL_ENV 'Scripts') + ';' + $env:Path
    if ($Install) {
        & $venvPython -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "L’installation Python a échoué." }
    }
    if ($Install -or -not (Test-Path -LiteralPath 'frontend\node_modules')) {
        Push-Location frontend
        try { npm.cmd ci --no-audit --no-fund; if ($LASTEXITCODE -ne 0) { throw "L’installation React a échoué." } }
        finally { Pop-Location }
        $Build = $true
    }
    if ($Build -or -not (Test-Path -LiteralPath 'frontend\dist\index.html')) {
        Push-Location frontend
        try { npm.cmd run build; if ($LASTEXITCODE -ne 0) { throw 'La compilation React a échoué.' } }
        finally { Pop-Location }
    }
    Write-Host "Neurovia : http://127.0.0.1:$Port" -ForegroundColor Cyan
    Write-Host "API : http://127.0.0.1:$Port/docs | Ctrl+C pour arrêter."
    if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$Port" }
    & $venvPython -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
}
finally { Pop-Location }
