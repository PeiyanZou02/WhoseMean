param(
    [int]$Limit = 200,
    [int]$Epochs = 30
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$secureKey = Read-Host 'Paste your Harvard Art Museums API key (it will not be saved)' -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $env:HARVARD_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    & "$PSScriptRoot/.venv/Scripts/python.exe" harvard_pipeline.py collect --limit $Limit
    if ($LASTEXITCODE -ne 0) { throw 'Collection failed.' }
    & "$PSScriptRoot/.venv/Scripts/python.exe" harvard_pipeline.py mean
    if ($LASTEXITCODE -ne 0) { throw 'Mean calculation failed.' }
    & "$PSScriptRoot/.venv/Scripts/python.exe" harvard_pipeline.py train --epochs $Epochs
    if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }
    Write-Host 'Harvard pilot finished. Restart the lab to display the real collection results.'
} finally {
    $env:HARVARD_API_KEY = $null
    if ($keyPointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer) }
}
