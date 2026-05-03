# AIDRA — Evaluator's Guide

> **For the SatCen technical evaluator.** This page maps each criterion of the tender (`SATCEN/2026/OP/0003` §7) to the concrete artefact that proves AIDRA satisfies it. Read in any order; every link is a 5-second click.

---

## TL;DR

| What you want to see | Open this |
|---|---|
| The system running | http://localhost:8000/api/health |
| The vessel detection map | http://localhost:3000 → `AIDRA — Detection Map` |
| The Tip & Cue autonomous re-tasking demo | http://localhost:3000 → `AIDRA — Tip & Cue Replay` |
| One AI-explained detection (Grad-CAM + CFAR) | [`D4_INTERPRETABILITY_ANNEX.md`](D4_INTERPRETABILITY_ANNEX.md) |
| The full evidence bundle (D3) | `docker exec aidra-app python -m src.traceability bundle --out /tmp/d3` |
| Verify a bundle offline | `docker exec aidra-app python -m src.traceability verify-bundle <path>` |
| Honest self-assessment vs the rubric | read this guide §6 |

---

## 1. Q3 — SAR methodology and simulation (10 pts)

| What we did | Where to look |
|---|---|
| Sentinel-1 GRD calibration σ⁰ → linear power | `src/pipeline/preprocessing.py:_parse_calibration_lut` |
| Lee speckle filter (7×7) | `src/pipeline/preprocessing.py:_lee_filter` |
| Edge-of-swath filter via footprint geometry + longitude clustering fallback (I-SAR-2) | `src/pipeline/engine.py:_save_detections` |
| Real-footprint clipping (no global-land-mask filtering, I-SAR-3) | `src/pipeline/preprocessing.py:_calculate_valid_footprint` |
| Quality gate `quality=invalid` if any pre-step missing (I-SAR-1) | `src/pipeline/preprocessing.py:_evaluate_scene_quality` + Prometheus counter `aidra_scenes_processed_total{quality}` |
| EPSG:4326 enforced everywhere (I-SAR-4) | grep `EPSG:4326` |
| SAR metadata persisted (`incidence_angle`, `polarisation`, `orbit_direction`, `relative_orbit`, `product_type`, `pixel_spacing`) | `src/pipeline/preprocessing.py:parse_sar_metadata` + DB column on `execution_log` |
| Constraint profiles (ground / sat-mid / sat-low / sat-extreme) | `src/profiles/definitions.py` and `src/profiles/manager.py` |

> **Honest gap (documented in scoring):** Range-Doppler terrain correction with DEM is scaffolded in `src/pipeline/terrain_correction.py` but **not yet wired in** — the production geo-referencing uses a linear affine over GCPs, which is acceptable for flat-sea AOI like Gibraltar. Constraint profiles only enforce RAM/CPU limits on Linux; on macOS dev hosts the system logs a warning at startup and runs without enforcement. Both gaps are tracked in `RISK_REGISTER.md`.

---

## 2. Q3 — GEOINT integration (10 pts)

| Capability | Endpoint |
|---|---|
| **STAC 1.0.0** Catalog + Collection + Items with `sar`/`sat`/`view` extensions | `GET /api/stac/catalog.json` |
| Items with rich SAR properties (`sar:product_type`, `sar:polarizations`, `sar:relative_orbit`, `sat:orbit_state`, `view:incidence_angle`) and AIDRA traceability (`aidra:image_hash`, `aidra:output_hash`, `aidra:input_params_hash`, `aidra:commit_sha`) | `GET /api/stac/collections/detections/items?limit=10` |
| Dynamic `extent.spatial.bbox` and `temporal.interval` (computed from real data) | look at `extent` in the Collection response |
| OGC API Features-style pagination (`numberMatched` + `numberReturned` + `next`/`prev` rel-links) | item list payload |
| Per-Item `assets.detections` filtered by `execution_id` (not by model) | open any Item, click the asset href |
| **GeoJSON** RFC 7946 with `application/geo+json` and on_land / cluster_anomaly filters | `GET /api/detections.geojson?on_land=false&cluster_anomaly=false&execution_id=<UUID>` |
| **PNG thumbnails** of the SAR crop around each detection | `GET /api/detections/{id}/thumbnail.png` |
| Ready for QGIS, ArcGIS, pystac, ogr2ogr (verified) | n/a |

> **Honest gap:** no WMS/MVT tile service yet. STAC Item Search and OGC API
> Features conformance endpoints are implemented; map-tile serving remains out
> of scope for this POC.

---

## 3. Q3 — Demo + traceability + AI documentation (20 pts, shared)

### 3.1 Traceability

| What | Where |
|---|---|
| `image_hash` + `model_hash` + `output_hash` + `input_params_hash` per run | `execution_log` table |
| `commit_sha` per run (build-arg → ENV → DB) | `execution_log.commit_sha` |
| `pending → success/error/invalid` state machine (failed runs are kept) | `execution_log.status` |
| `run_id` propagated to logs (Loki) | every Loki log line carries `execution_id` |
| Migration history | `src/db/migrations/00{1..5}_*.sql` |
| **D3 bundler** packs everything for offline audit | `docker exec aidra-app python -m src.traceability bundle --out /tmp/d3 --no-archive` |
| Bundle includes: `executions.csv`, `detections.csv`, `detections.geojson`, `settings.json` (secrets redacted), `models/*.MODEL_CARD.md` (matched by name and by `model_hash`), `prometheus_snapshot.txt`, `thumbnails/*.png`, `MANIFEST.json` (per-file SHA256 + `settings_hash` + `commit_sha`), and `MANIFEST.sha256` (root signature) | inspect `/tmp/d3/d3-<timestamp>/` |
| **Offline bundle verifier** | `docker exec aidra-app python -m src.traceability verify-bundle /tmp/d3/d3-<timestamp>` → `Result: PASS` |

### 3.2 AI Act conformity

| Item | Where |
|---|---|
| **Classification + base legal + human oversight** (no Anexo III argument) | [`AI_ACT_DECLARATION.md`](AI_ACT_DECLARATION.md) (1 page) |
| **MODEL_CARD per registered model** | `models/cards/*.MODEL_CARD.md` (5 cards) |
| **Gate**: no MODEL_CARD → no registration (no silent fallback) | `src/models/manager.py:_require_model_card` + `tests/test_invariants.py:TestIAIA1AICardGate` |
| **Validation metrics** on the primary YOLO and CFAR cards (mAP, Pd, FAR/km2, precision) | `models/cards/vesseltracker-sar-yolov8.MODEL_CARD.md` and `models/cards/cfar-default.MODEL_CARD.md` § D2 oficial |
| **Interpretability D4 annex**: 20 × {Grad-CAM, CFAR score map} on real detections, with manifest (commit_sha + model_hash + per-PNG SHA256) | [`D4_INTERPRETABILITY_ANNEX.md`](D4_INTERPRETABILITY_ANNEX.md) + `/data/interpretability/<run>/` |
| Reproducible: `docker exec aidra-app python /app/scripts/run_interpretability.py --n 20` | n/a |

> **Honest gap:** validation is real but still geographically narrow: 11
> Mediterranean / Adriatic xView3-SAR validation scenes, not a global SAR
> benchmark and not a Strait-of-Gibraltar-specific labelled set. The model cards
> document this as a lower-bound transfer measurement.

---

## 4. Q2 — Project management (15 pts)

| Item | Where |
|---|---|
| Plan operativo de despliegue (OCI ARM A1 Free Tier, fases) | `mvp_oci.md` (477 lines) |
| Especificación técnica completa | `TECHNICAL_SPEC.md` |
| Engineering operating notes (gates, invariants, anti-patterns) | `CLAUDE.md` |
| **Risk register** con 7 riesgos, severidad/probabilidad/mitigación/trigger/plan B | [`RISK_REGISTER.md`](RISK_REGISTER.md) |
| Tests automáticos de invariantes | `pytest -k invariant`; full suite: `pytest -q` (245 tests) |

---

## 5. WOW effects (extras del pliego)

| Feature | Where to see it |
|---|---|
| **SAR thumbnail per detection** (visual proof of every vessel) | `AIDRA — Detection Map` → `Detection gallery` panel |
| **Tip & Cue autonomous re-tasking replay** (T0 → Cue → T1 timeline with thumbnails before/after) | `AIDRA — Tip & Cue Replay` → click `▶ Open replay` on any row |

---

## 6. Self-score against the rubric (honest)

Current public self-score after the D2 validation, STAC/OGC work,
traceability hardening and interpretability annex:

```
Q1 Equipo:                            n/a (proyecto personal)
Q2 Plan + Riesgos:                    12/15
Q3 Metodología SAR:                    5/10  (TC dead code, profiles macOS-noop)
Q3 Integración GEOINT:                 9/10  (no WMS/MVT tile service)
Q3 Demo + trazabilidad + IA:          19/20  (validation scope is narrow)
                                      -----
TOTAL Q-técnico (Q2+Q3):              45/55  (banda 80%+, "muy buena")
                                              mínimo para pasar = 30/55
```

Top remaining levers if more time: SAR TC real (+1), STAC API search (+1.5), AI Act tests + métricas formales (+1.5).

---

## 7. Reproduce the demo locally

```bash
git clone <repo>
cd AIDRA
cp .env.example .env  # fill COPERNICUS_USER, COPERNICUS_PASSWORD, DB_PASSWORD, GRAFANA_PASSWORD
AIDRA_COMMIT_SHA="$(git rev-parse HEAD)" docker compose up -d
docker exec -i aidra-db psql -U aidra -d aidra < src/db/migrations/001_init.sql
docker exec -i aidra-db psql -U aidra -d aidra < src/db/migrations/002_indexes.sql
docker exec -i aidra-db psql -U aidra -d aidra < src/db/migrations/003_tipcue.sql
docker exec -i aidra-db psql -U aidra -d aidra < src/db/migrations/004_traceability.sql
docker exec -i aidra-db psql -U aidra -d aidra < src/db/migrations/005_thumbnails.sql

# Trigger one pipeline run on Gibraltar:
curl -X POST http://localhost:8000/api/pipeline/trigger \
  -H "Content-Type: application/json" \
  -d '{"zone":"gibraltar","model":"vesseltracker-sar-yolov8","profile":"ground"}'

# Watch the result on Grafana:
open http://localhost:3000

# Build the D3 evidence bundle:
docker exec aidra-app python -m src.traceability bundle --out /tmp/d3 --no-archive
docker exec aidra-app python -m src.traceability verify-bundle /tmp/d3/d3-<timestamp>
```

---

*Last updated: 2026-04-26. Generated commit: `0cc8ccd`.*
