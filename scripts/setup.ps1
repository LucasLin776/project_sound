param([string]$EnvName = 'wind-bolt-audio', [switch]$SkipWeights)
$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDir
try {
    if ($EnvName -eq 'base' -or $EnvName -notmatch '^[A-Za-z0-9_-]+$') {
        throw 'Choose a dedicated Conda environment name (not base).'
    }
    $condaCommand = Get-Command conda.exe -ErrorAction SilentlyContinue
    if (-not $condaCommand) { $condaCommand = Get-Command conda -ErrorAction Stop }
    $condaExecutable = $condaCommand.Source
    if (-not $condaExecutable) { $condaExecutable = 'conda' }
    $envInfoText = & $condaExecutable env list --json
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read Conda environments.' }
    $envInfo = ($envInfoText -join "`n") | ConvertFrom-Json
    $matchingEnvs = @($envInfo.envs | Where-Object { (Split-Path -Leaf $_) -eq $EnvName })
    if ($matchingEnvs.Count -eq 0) {
        & $condaExecutable env create --name $EnvName --file environment.yml --yes
        if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed.' }
    }
    & $condaExecutable run --no-capture-output -n $EnvName python -c 'import sys; assert sys.version_info[:2] == (3, 12), "This project requires Python 3.12"'
    if ($LASTEXITCODE -ne 0) { throw 'The Conda environment must use Python 3.12.' }
    & $condaExecutable run --no-capture-output -n $EnvName python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'Torch installation failed.' }
    if (Test-Path -LiteralPath 'requirements.lock.windows.txt') {
        & $condaExecutable run --no-capture-output -n $EnvName python -m pip install -r requirements.lock.windows.txt
        if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed.' }
    }
    & $condaExecutable run --no-capture-output -n $EnvName python -m pip install -e '.[api,dev]'
    if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
    if (-not $SkipWeights) {
        & $condaExecutable run --no-capture-output -n $EnvName python scripts/download_checkpoints.py
        if ($LASTEXITCODE -ne 0) { throw 'Weight verification failed.' }
    }
    & $condaExecutable run --no-capture-output -n $EnvName python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
    Write-Output "Ready. Run: conda activate $EnvName"
} finally { Pop-Location }
