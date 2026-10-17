# Deploy API + monitoring to the current kubectl context (Docker Desktop or Minikube).
# Usage:  .\scripts\k8s_deploy.ps1            (with Prometheus + Grafana)
#         .\scripts\k8s_deploy.ps1 -NoMonitoring
param([switch]$NoMonitoring)

$ErrorActionPreference = "Stop"
# Native-command pipes (kubectl ... | kubectl apply -f -) should pass UTF-8, not ASCII/UTF-16.
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$root = Split-Path -Parent $PSScriptRoot
$image = if ($env:IMAGE) { $env:IMAGE } else { "heart-risk-api:latest" }

function Kube {
    kubectl @args
    if ($LASTEXITCODE -ne 0) { throw "kubectl $($args -join ' ') failed (exit $LASTEXITCODE)" }
}

$ctx = kubectl config current-context
Write-Host ">> kubectl context: $ctx" -ForegroundColor Cyan
if ($ctx -like "*minikube*") {
    Write-Host ">> Loading $image into minikube"
    minikube image load $image
}

Kube apply -f "$root\k8s\namespace.yaml"
Kube apply -f "$root\k8s\configmap.yaml"
Kube apply -f "$root\k8s\deployment.yaml" -f "$root\k8s\service.yaml"
kubectl apply -f "$root\k8s\hpa.yaml"
if ($LASTEXITCODE -ne 0) { Write-Host "(HPA skipped: metrics API unavailable)" }

if (-not $NoMonitoring) {
    Write-Host ">> Monitoring stack" -ForegroundColor Cyan
    $maps = @{
        "grafana-datasources"        = "$root\monitoring\grafana\provisioning\datasources\prometheus.yml"
        "grafana-dashboard-provider" = "$root\monitoring\grafana\provisioning\dashboards\provider.yml"
        "grafana-dashboards"         = "$root\monitoring\grafana\dashboards\heart-api.json"
    }
    foreach ($name in $maps.Keys) {
        kubectl -n heart-risk create configmap $name --from-file="$($maps[$name])" `
            --dry-run=client -o yaml | kubectl apply -f -
        if ($LASTEXITCODE -ne 0) { throw "creating configmap $name failed" }
    }
    Kube apply -f "$root\k8s\monitoring\prometheus.yaml" -f "$root\k8s\monitoring\grafana.yaml"
}

Write-Host ">> Waiting for rollout" -ForegroundColor Cyan
Kube -n heart-risk rollout status deployment/heart-api --timeout=180s
Kube -n heart-risk get "pods,svc,hpa" -o wide

Write-Host @"

Next steps
  Docker Desktop : API at http://localhost   (LoadBalancer on port 80)
  Minikube       : run 'minikube tunnel' in another window, use the EXTERNAL-IP above
  Quick access   : kubectl -n heart-risk port-forward svc/heart-api 8080:80
  Smoke test     : .\scripts\smoke_test.ps1 -BaseUrl http://localhost
  Grafana        : http://localhost:30300 (admin/admin)   Prometheus: http://localhost:30090
"@
