# Smoke-test a running API from PowerShell (Windows PowerShell 5.1 or PowerShell 7).
# Usage:  .\scripts\smoke_test.ps1                      (defaults to http://localhost:8000)
#         .\scripts\smoke_test.ps1 -BaseUrl http://localhost
param([string]$BaseUrl = "http://localhost:8000")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "Waiting for $BaseUrl/health ..."
$healthy = $false
for ($i = 1; $i -le 30; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "$BaseUrl/health" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $healthy = $true; break }
    } catch { }
    Start-Sleep -Seconds 2
}
if (-not $healthy) { Write-Host "API did not become healthy" -ForegroundColor Red; exit 1 }

Write-Host "`n--- GET /health" -ForegroundColor Cyan
Invoke-RestMethod "$BaseUrl/health" | ConvertTo-Json

Write-Host "`n--- GET /model-info" -ForegroundColor Cyan
Invoke-RestMethod "$BaseUrl/model-info" | ConvertTo-Json -Depth 6

Write-Host "`n--- POST /predict" -ForegroundColor Cyan
$body = Get-Content -Raw (Join-Path $root "sample_request.json")
$resp = Invoke-RestMethod -Method Post -Uri "$BaseUrl/predict" -ContentType "application/json" -Body $body
$resp | ConvertTo-Json
if (($resp.prediction -notin 0, 1) -or ($resp.confidence -lt 0) -or ($resp.confidence -gt 1)) {
    Write-Host "Unexpected prediction response" -ForegroundColor Red; exit 1
}
Write-Host "prediction OK" -ForegroundColor Green

Write-Host "`n--- POST /predict with invalid input (expect 422)" -ForegroundColor Cyan
$code = 0
try {
    Invoke-WebRequest -Method Post -Uri "$BaseUrl/predict" -ContentType "application/json" `
        -Body '{"age": 5}' -UseBasicParsing | Out-Null
    $code = 200
} catch {
    $code = [int]$_.Exception.Response.StatusCode
}
if ($code -ne 422) { Write-Host "expected 422, got $code" -ForegroundColor Red; exit 1 }
Write-Host "validation OK" -ForegroundColor Green

Write-Host "`n--- GET /metrics (excerpt)" -ForegroundColor Cyan
(Invoke-WebRequest "$BaseUrl/metrics" -UseBasicParsing).Content -split "`n" |
    Select-String -Pattern "^heart_api_(requests|predictions)_total" |
    Select-Object -First 10 | ForEach-Object { $_.Line }

Write-Host "`nAll smoke tests passed." -ForegroundColor Green
