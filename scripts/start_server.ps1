param([int]$Port = 8000, [string]$ConfigPath = '', [string]$EnvName = 'wind-bolt-audio')
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDir
try {
    $arguments = @('-m', 'bolt_audio')
    if ($ConfigPath) { $arguments += @('--config', $ConfigPath) }
    $arguments += @('serve', '--host', '127.0.0.1', '--port', $Port)
    & conda run --no-capture-output -n $EnvName python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Server failed.' }
} finally { Pop-Location }
