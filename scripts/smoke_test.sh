#!/usr/bin/env bash
# Smoke-test a running API. Usage: scripts/smoke_test.sh [BASE_URL]
# Exits non-zero if any check fails, so it can be used in CI.
set -euo pipefail
BASE_URL="${1:-http://localhost:8000}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "Waiting for ${BASE_URL}/health ..."
for i in $(seq 1 30); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then break; fi
  sleep 2
  if [ "$i" -eq 30 ]; then echo "API did not become healthy"; exit 1; fi
done

echo "--- GET /health"
curl -fsS "${BASE_URL}/health"; echo

echo "--- GET /model-info"
curl -fsS "${BASE_URL}/model-info"; echo

echo "--- POST /predict"
RESPONSE=$(curl -fsS -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" -d @"${HERE}/sample_request.json")
echo "${RESPONSE}"
echo "${RESPONSE}" | python -c "import json,sys; r=json.load(sys.stdin); assert r['prediction'] in (0,1) and 0<=r['confidence']<=1, r; print('prediction OK')"

echo "--- POST /predict with invalid input (expect 422)"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" -d '{"age": 5}')
[ "${CODE}" = "422" ] && echo "validation OK" || { echo "expected 422, got ${CODE}"; exit 1; }

echo "--- GET /metrics (excerpt)"
curl -fsS "${BASE_URL}/metrics" | grep -E "^heart_api_(requests|predictions)_total" | head -n 10
echo "All smoke tests passed."
