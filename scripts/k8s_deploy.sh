#!/usr/bin/env bash
# Deploy API + monitoring to the current kubectl context (Minikube / Docker Desktop / cloud).
# Usage: scripts/k8s_deploy.sh [--no-monitoring]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-heart-risk-api:latest}"

if kubectl config current-context 2>/dev/null | grep -q minikube; then
  echo ">> Loading ${IMAGE} into minikube"
  minikube image load "${IMAGE}"
fi

kubectl apply -f "${ROOT}/k8s/namespace.yaml"
kubectl apply -f "${ROOT}/k8s/configmap.yaml"
kubectl apply -f "${ROOT}/k8s/deployment.yaml" -f "${ROOT}/k8s/service.yaml"
kubectl apply -f "${ROOT}/k8s/hpa.yaml" || echo "(HPA skipped: metrics API unavailable)"

if [ "${1:-}" != "--no-monitoring" ]; then
  echo ">> Monitoring stack"
  kubectl -n heart-risk create configmap grafana-datasources \
    --from-file="${ROOT}/monitoring/grafana/provisioning/datasources/prometheus.yml" \
    --dry-run=client -o yaml | kubectl apply -f -
  kubectl -n heart-risk create configmap grafana-dashboard-provider \
    --from-file="${ROOT}/monitoring/grafana/provisioning/dashboards/provider.yml" \
    --dry-run=client -o yaml | kubectl apply -f -
  kubectl -n heart-risk create configmap grafana-dashboards \
    --from-file="${ROOT}/monitoring/grafana/dashboards/heart-api.json" \
    --dry-run=client -o yaml | kubectl apply -f -
  kubectl apply -f "${ROOT}/k8s/monitoring/prometheus.yaml" -f "${ROOT}/k8s/monitoring/grafana.yaml"
fi

echo ">> Waiting for rollout"
kubectl -n heart-risk rollout status deployment/heart-api --timeout=180s
kubectl -n heart-risk get pods,svc,hpa -o wide
cat <<MSG

Next steps
  Minikube LoadBalancer : run 'minikube tunnel', then use the EXTERNAL-IP above
  Quick access          : kubectl -n heart-risk port-forward svc/heart-api 8080:80
  Smoke test            : scripts/smoke_test.sh http://localhost:8080
  Prometheus / Grafana  : minikube service -n heart-risk prometheus  |  grafana (admin/admin)
MSG
