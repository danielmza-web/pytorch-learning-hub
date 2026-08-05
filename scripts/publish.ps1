[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$Message
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$netlifySiteId = 'f58cc486-94eb-4664-ba33-ae59491a53bc'

function Invoke-CheckedCommand {
    param([scriptblock]$Command)

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE."
    }
}

Set-Location $repositoryRoot

Invoke-CheckedCommand { python -m mkdocs build --strict }
Invoke-CheckedCommand { python tests/validate_content.py }
Invoke-CheckedCommand { python -c "import torch; print(torch.__version__)" }
Invoke-CheckedCommand { python tests/smoke_examples.py }

if ($PSCmdlet.ShouldProcess('pytorch.dazu.xyz', "Publish: $Message")) {
    Invoke-CheckedCommand {
        netlify deploy --prod --site $netlifySiteId --dir=site --no-build --message $Message
    }
}
