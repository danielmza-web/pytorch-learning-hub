[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot

function Invoke-CheckedCommand {
    param([scriptblock]$Command)

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE."
    }
}

Set-Location $repositoryRoot
$localPython = Join-Path $repositoryRoot '.venv/Scripts/python.exe'
if (Test-Path -LiteralPath $localPython) {
    Invoke-CheckedCommand { & $localPython -m mkdocs serve }
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    Invoke-CheckedCommand { py -3.14 -m mkdocs serve }
} else {
    Invoke-CheckedCommand { python -m mkdocs serve }
}
