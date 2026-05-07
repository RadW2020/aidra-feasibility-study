#!/usr/bin/env bash
set -euo pipefail

base_url="${GRAFANA_BASE_URL:-https://aidra.uliber.com}"
out_dir="${1:-.tmp_dash_audit/screenshots}"
from="${GRAFANA_CAPTURE_FROM:-now-30d}"
to="${GRAFANA_CAPTURE_TO:-now}"
width="${GRAFANA_CAPTURE_WIDTH:-1800}"
height="${GRAFANA_CAPTURE_HEIGHT:-1200}"
tz="${GRAFANA_CAPTURE_TZ:-Europe%2FMadrid}"

mkdir -p "$out_dir"

dashboards=(
  aidra-home
  aidra-evaluator-evidence
  aidra-map-detections
  aidra-pipeline-metrics
  aidra-compression-bench
  aidra-constraint-profiles
  aidra-traceability
  aidra-obdp-value
  aidra-orbital-latency
  aidra-orbital-resilience
  aidra-tipcue-replay
)

for uid in "${dashboards[@]}"; do
  url="${base_url}/render/d/${uid}/capture?orgId=1&from=${from}&to=${to}&width=${width}&height=${height}&tz=${tz}&kiosk"
  curl -fsS "$url" -o "${out_dir}/${uid}.png"
done
