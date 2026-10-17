# Send varied requests so the Grafana panels have something to show.
# About 10% are deliberately invalid (422s), so the error panels are not empty.
# Usage:  .\scripts\generate_traffic.ps1 -Count 300 [-BaseUrl http://localhost]
param([string]$BaseUrl = "http://localhost:8000", [int]$Count = 200)

$ok = 0; $invalid = 0
for ($i = 1; $i -le $Count; $i++) {
    $patient = [ordered]@{
        age      = Get-Random -Minimum 35 -Maximum 76
        sex      = Get-Random -InputObject 0, 1
        cp       = Get-Random -InputObject 1, 2, 3, 4
        trestbps = Get-Random -Minimum 100 -Maximum 180
        chol     = Get-Random -Minimum 160 -Maximum 330
        fbs      = Get-Random -InputObject 0, 0, 0, 1
        restecg  = Get-Random -InputObject 0, 1, 2
        thalach  = Get-Random -Minimum 100 -Maximum 195
        exang    = Get-Random -InputObject 0, 1
        oldpeak  = [math]::Round((Get-Random -Minimum 0 -Maximum 40) / 10, 1)
        slope    = Get-Random -InputObject 1, 2, 3
        ca       = Get-Random -InputObject 0, 0, 1, 2, 3
        thal     = Get-Random -InputObject 3, 3, 6, 7
    }
    if ((Get-Random -Minimum 0 -Maximum 10) -eq 0) { $patient.age = 5 }  # invalid on purpose
    $json = $patient | ConvertTo-Json -Compress
    try {
        Invoke-RestMethod -Method Post -Uri "$BaseUrl/predict" -ContentType "application/json" `
            -Body $json | Out-Null
        $ok++
    } catch { $invalid++ }
    if ($i % 50 -eq 0) { Write-Host "$i requests sent ($ok ok, $invalid rejected)" }
}
Write-Host "Done: $ok predictions, $invalid rejected requests." -ForegroundColor Green
