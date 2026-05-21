param(
    [int]$BackendPort = 7860,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BackendScript = Join-Path $PSScriptRoot "start-backend.ps1"
$FrontendScript = Join-Path $PSScriptRoot "start-frontend.ps1"
$BackendLog = Join-Path $Root "backend.log"
$BackendErrorLog = Join-Path $Root "backend.err.log"
$FrontendLog = Join-Path $Root "frontend.log"
$FrontendErrorLog = Join-Path $Root "frontend.err.log"

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

Stop-PortProcess -TargetPort $BackendPort
Stop-PortProcess -TargetPort $FrontendPort

Write-Host "Starting backend on http://127.0.0.1:$BackendPort..."
$backend = Start-Process powershell -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $BackendScript,
    "-Port", $BackendPort
) -WorkingDirectory $Root -RedirectStandardOutput $BackendLog -RedirectStandardError $BackendErrorLog -WindowStyle Hidden -PassThru

$healthUrl = "http://127.0.0.1:$BackendPort/health"
$backendReady = $false
for ($attempt = 1; $attempt -le 30; $attempt++) {
    Start-Sleep -Seconds 1
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        if ($health.status -eq "ok") {
            $backendReady = $true
            break
        }
    } catch {
        if ($backend.HasExited) {
            throw "Backend exited before health check passed. See $BackendErrorLog."
        }
    }
}

if (-not $backendReady) {
    throw "Backend did not become healthy at $healthUrl. See $BackendErrorLog."
}

Write-Host "Backend healthy at $healthUrl"
Write-Host "Starting frontend on http://127.0.0.1:$FrontendPort..."
$frontend = Start-Process powershell -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $FrontendScript,
    "-Port", $FrontendPort
) -WorkingDirectory (Join-Path $Root "frontend") -RedirectStandardOutput $FrontendLog -RedirectStandardError $FrontendErrorLog -WindowStyle Hidden -PassThru

$frontendUrl = "http://127.0.0.1:$FrontendPort"
$frontendReady = $false
for ($attempt = 1; $attempt -le 30; $attempt++) {
    Start-Sleep -Seconds 1
    try {
        $response = Invoke-WebRequest -Uri $frontendUrl -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            $frontendReady = $true
            break
        }
    } catch {
        if ($frontend.HasExited) {
            throw "Frontend exited before it became reachable. See $FrontendErrorLog."
        }
    }
}

if (-not $frontendReady) {
    throw "Frontend did not become reachable at $frontendUrl. See $FrontendErrorLog."
}

Write-Host "CredibleX is starting:"
Write-Host "  Backend:  http://127.0.0.1:$BackendPort"
Write-Host "  Frontend: $frontendUrl"
Write-Host "  Backend PID:  $($backend.Id)"
Write-Host "  Frontend PID: $($frontend.Id)"
