param(
    [switch]$NoOpen
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$url = 'http://127.0.0.1:8765/'
try {
    $health = Invoke-RestMethod "$url/api/health" -TimeoutSec 2
    if ($health.status -eq 'ok') {
        Write-Host "Whose Mean is already running at $url"
        if (-not $NoOpen) { Start-Process $url }
        exit 0
    }
} catch {}

$process = Start-Process -FilePath "$PSScriptRoot/.venv/Scripts/python.exe" `
    -ArgumentList 'lab.py','serve' -WorkingDirectory $PSScriptRoot `
    -WindowStyle Hidden -PassThru

for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 250
    try {
        $health = Invoke-RestMethod "$url/api/health" -TimeoutSec 1
        if ($health.status -eq 'ok') {
            Write-Host "Whose Mean started (PID $($process.Id)) at $url"
            if (-not $NoOpen) { Start-Process $url }
            exit 0
        }
    } catch {}
}

throw "Whose Mean did not start. Run .venv/Scripts/python.exe lab.py serve to see the error."
