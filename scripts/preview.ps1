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
Invoke-CheckedCommand { python -m pip install -r requirements.txt }
Invoke-CheckedCommand { python -m mkdocs serve }
