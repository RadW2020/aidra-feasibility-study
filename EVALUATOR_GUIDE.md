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
| The full evidence bundle (D3) | `POST /api/traceability/bundle` (bearer token) — see `EVIDENCE.md`; latest bundle mirrored under `evidence_bundles/` |
| Verify a bundle offline | `python -m src.traceability verify-bundle <extracted-bundle-dir>` on the downloaded tarball |
| Detection quality against real ground truth (D2) | `reports/validation_xview3_adriatic_full_vessels_*.json` + README table |
| Honest self-assessment vs the rubric | read this guide §6 |

---

## 1. Q3 — SAR methodology and simulation (10 pts)

| What we did | Where to look |
|---|---|
| Sentinel-1 GRD calibration σ⁰ → linear power | `src/pipeline/preprocessing.py:_parse_calibration_lut` |
| Lee speckle filter (7×7) | `src/pipeline/preprocessing.py:_lee_filter` |
| Edge-of-swath filter via footprint geometry + longitude clustering fallback (I-SAR-2) | `src/pipeline/engine.py:_save_detections` |
| Real-footprint clipping (I-SAR-3) | `src/pipeline/preprocessing.py:_calculate_valid_footprint` |
| `global-land-mask` used as a **pre-inference sea mask for CFAR** (rotation-aware affine) and as the informational `on_land` flag; never to discard a produced detection (I-SAR-3 / I-DET-2) | `src/pipeline/detection.py:_build_sea_mask_from_affine`, `src/pipeline/engine.py:_save_detections` |
| Quality gate `quality=invalid` if any pre-step missing (I-SAR-1) | `src/pipeline/preprocessing.py:_evaluate_scene_quality` + Prometheus counter `aidra_scenes_processed_total{quality}` |
| EPSG:4326 enforced everywhere (I-SAR-4) | grep `EPSG:4326` |
| SAR metadata persisted (`incidence_angle`, `polarisation`, `orbit_direction`, `relative_orbit`, `product_type`, `pixel_spacing`) | `src/pipeline/preprocessing.py:parse_sar_metadata` + DB column on `execution_log` |
| Constraint profiles (ground / sat-high / sat-mid / sat-low / sat-extreme) | `src/profiles/definitions.py` and `src/profiles/manager.py` |
| Validation harness that runs the production `DetectionEngine` on labelled xView3 scenes and anchors every report to `commit_sha` / `model_hash` / `settings_hash` / seed | `src/validation/harness.py`, `scripts/validate_xview3_serial.py` |

> **Honest gap (documented in scoring):** Range-Doppler terrain correction with DEM is scaffolded in `src/pipeline/terrain_correction.py` but **not yet wired in** — the production geo-referencing uses a linear affine over GCPs, which is acceptable for flat-sea AOI like Gibraltar. Constraint profiles enforce **CPU** only (affinity + duty-cycle throttle, Linux only); **RAM budgets are measured, not enforced** — peak RSS above the budget is recorded in the run's `notes` (R6/R9 context). Both gaps are tracked in `RISK_REGISTER.md`.

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
| **High-precision tier** — detections where CFAR and YOLO agreed (`source=fused`; precision 0.31 vs 0.16, FAR 8× lower, Pd 0.12 vs 0.36 on xView3); `tier` property on every feature | `GET /api/detections.geojson?tier=high`, `GET /api/ogc/collections/detections/items?source=fused`, Grafana *Source / tier* selector |
| **PNG thumbnails** of the SAR crop around each detection | `GET /api/detections/{id}/thumbnail.png` |
| Ready for QGIS, ArcGIS, pystac, ogr2ogr (verified) | n/a |

> **Honest gap:** no WMS/MVT tile service yet. STAC Item Search
> (`POST /api/stac/search`) and OGC API Features conformance endpoints are
> implemented; map-tile serving remains out of scope for this POC.

---

## 3. Q3 — Demo + traceability + AI documentation (20 pts, shared)

### 3.1 Traceability

| What | Where |
|---|---|
| `image_hash` + `model_hash` + `output_hash` + `input_params_hash` per run | `execution_log` table |
| `commit_sha` per run (build-arg → ENV → DB) | `execution_log.commit_sha` |
| `pending → success/error/invalid` state machine (failed runs are kept) | `execution_log.status` |
| `run_id` propagated to logs (Loki) | every Loki log line carries `execution_id` |
| Migration history | `src/db/migrations/001..018_*.sql`, applied idempotently by `Database.run_migrations` (`_migrations` table) |
| **D3 bundler** packs everything for offline audit | `POST /api/traceability/bundle` (same orchestration as `scripts/build_d3_bundle.py`) |
| Bundle includes: `executions.csv`, `detections.csv`, `detections.geojson`, `settings.json` (secrets redacted), `models/*.MODEL_CARD.md` (matched by name and by `model_hash`), `prometheus_snapshot.txt`, `thumbnails/*.png`, `MANIFEST.json` (per-file SHA256 + `settings_hash` + `commit_sha`), and `MANIFEST.sha256` (root signature) | inspect `/tmp/d3/d3-<timestamp>/` |
| **Offline bundle verifier** | `python -m src.traceability verify-bundle <extracted dir>` → `Result: PASS` (last run: 22 243/22 243 files OK, see `EVIDENCE.md`) |
| **Validation reports** carry `commit_sha`, `model_hash`, `settings_hash`, seed and per-scene input SHA256; `POST /api/validation/import` refuses a report without `commit_sha` | `reports/*.json` → `provenance`, migration 018 |

### 3.2 AI Act conformity

| Item | Where |
|---|---|
| **Classification + base legal + human oversight** (no Anexo III argument) | [`AI_ACT_DECLARATION.md`](AI_ACT_DECLARATION.md) (1 page) |
| **MODEL_CARD per registered model** | `models/cards/*.MODEL_CARD.md` (5 cards) |
| **Gate**: no MODEL_CARD → no registration (no silent fallback) | `src/models/manager.py:_require_model_card` + `tests/test_invariants.py:TestIAIA1AICardGate` |
| **Validation metrics** on the primary YOLO and CFAR cards (AP, Pd, FAR/km², precision, F1; raw and sea-only) | `models/cards/vesseltracker-sar-yolov8.MODEL_CARD.md` and `models/cards/cfar-default.MODEL_CARD.md` § *D2 — pipeline completo (2026-08-28)* |
| **Interpretability D4 annex**: 20 × {Grad-CAM, CFAR score map} on real detections, with manifest (commit_sha + model_hash + per-PNG SHA256) | [`D4_INTERPRETABILITY_ANNEX.md`](D4_INTERPRETABILITY_ANNEX.md) + `/data/interpretability/<run>/` |
| Reproducible: `POST /api/interpretability/run` `{"n_samples": 20}` (same orchestration as `scripts/run_interpretability.py`) | n/a |

> **Honest gap:** validation is real (1 997 xView3 vessels, production
> `DetectionEngine`, identical settings for every detector) but geographically
> narrow: 11 Adriatic scenes from one track, VH only — not a global SAR
> benchmark and not a Strait-of-Gibraltar labelled set. It also exposed two
> design defects that are being re-validated: the IoU-based CFAR∩YOLO fusion
> never fired (R14) and the Lee filter cost YOLO 35 % of its recall (R15).

---

## 4. Q2 — Project management (15 pts)

| Item | Where |
|---|---|
| Plan operativo de despliegue (OCI ARM A1 Free Tier, fases) | `mvp_oci.md` (477 lines) |
| Especificación técnica completa | `TECHNICAL_SPEC.md` |
| Engineering operating notes (gates, invariants, anti-patterns) | `CLAUDE.md` |
| **Risk register** con 15 riesgos (R1–R15), severidad/probabilidad/mitigación/trigger/plan B; los fallos de auditoría propios (R6, R9, R11, R12, R13) están registrados, no borrados | [`RISK_REGISTER.md`](RISK_REGISTER.md) |
| Tests automáticos de invariantes | `pytest -k invariant`; full suite: `pytest -q` (499 tests) |

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

Top remaining levers if more time: SAR TC real (+1), real RAM enforcement in the constraint profiles (+1), a complete compression triplet with ΔmAP on the same scenes (+1.5).

---

## 7. Reproduce the demo locally

```bash
git clone <repo>
cd AIDRA
cp .env.example .env  # fill COPERNICUS_USER, COPERNICUS_PASSWORD, DB_PASSWORD, GRAFANA_PASSWORD
AIDRA_COMMIT_SHA="$(git rev-parse HEAD)" docker compose up -d
# the app applies the 18 migrations itself at startup (Database.run_migrations)

# Trigger one pipeline run on Gibraltar:
curl -X POST http://localhost:8000/api/pipeline/trigger \
  -H "Content-Type: application/json" \
  -d '{"zone":"gibraltar","model":"vesseltracker-sar-yolov8","profile":"ground"}'

# Watch the result on Grafana:
open http://localhost:3000

# Build the D3 evidence bundle through the API, then verify the download locally:
curl -X POST http://localhost:8000/api/traceability/bundle -H "Authorization: Bearer $AIDRA_API_TOKEN" \
  -H "Content-Type: application/json" -d '{"out_dir": "/data/interpretability/d3_bundles", "archive": true}'
python -m src.traceability verify-bundle <extracted bundle dir>
```

---

*Last updated: 2026-08-29 (Phase 0–2 of the 2026-08-28 improvement plan). Commit: see `git log -1 -- EVALUATOR_GUIDE.md`.*
