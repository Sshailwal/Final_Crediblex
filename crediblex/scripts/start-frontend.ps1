param(
    [int]$Port = 5173
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$FrontendRoot = Join-Path $Root "frontend"

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

Write-Host "Starting CredibleX frontend on http://localhost:$Port"
Set-Location $FrontendRoot
npm run dev -- --host 127.0.0.1 --port $Port
