param([string]$EnvName = 'wind-bolt-audio')
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDir
try {
    & conda run --no-capture-output -n $EnvName python -m bolt_audio infer --audio data/reference/bolt_drop.wav --output results/reference/inference.json
    if ($LASTEXITCODE -ne 0) { throw 'Inference failed.' }
} finally { Pop-Location }
