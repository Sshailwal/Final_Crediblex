param(
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

function Stop-PortProcess {
    param([int]$TargetPort)

    $connections = Get-NetTCPConnection -LocalPort $TargetPort -ErrorAction SilentlyContinue
    $processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique

    foreach ($processId in $processIds) {
        if ($processId -and $processId -ne $PID) {
            Write-Host "Stopping process $processId on port $TargetPort..."
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    }
}

Stop-PortProcess -TargetPort $Port

Write-Host "Starting CredibleX backend on http://127.0.0.1:$Port"
Set-Location $Root
python -m uvicorn api:app --host 127.0.0.1 --port $Port
