# AIDRA — In-orbit Vessel Detection POC

> **Artificial Intelligence In-orbit Data pRocessing Assessment** —
> Sentinel-1 SAR vessel detection running on commodity hardware
> that simulates the resource envelope of an on-board satellite
> processor.
>
> Inspired by EU SatCen tender `SATCEN/2026/OP/0003`. **Not** an
> operational product: an evaluation study on whether on-board AI
> for vessel detection is technically feasible under
> space-grade constraints.

[![CI](https://github.com/RadW2020/aidra-feasibility-study/actions/workflows/ci.yml/badge.svg)](https://github.com/RadW2020/aidra-feasibility-study/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## What this repo is

A working end-to-end pipeline that:

1. **Ingests** Sentinel-1 GRD scenes from Copernicus Data Space.
2. **Pre-processes** them (orbit-corrected calibration, Lee speckle
   filter, tiling, valid-footprint clipping).
3. **Detects vessels** with a CFAR + YOLOv8 ensemble.
4. **Persists** every detection in PostGIS with full provenance
   (SHA256 of image / model / output, run UUID, commit SHA, input
   params hash).
5. **Exposes** results through GeoJSON RFC 7946, STAC 1.0.0
   (with `sar`/`sat`/`view` extensions and `POST /stac/search`)
   and OGC API Features Part 1 — drop-in for QGIS, EODAG,
   pystac-client and SatCen-style GEOINT systems.
6. **Measures** detector performance under five hardware profiles
   (`ground`, `sat-high`, `sat-mid`, `sat-low`, `sat-extreme`)
   with energy estimates and p95 latency.
7. **Bundles** an evidence package (`MANIFEST.json` + per-file
   SHA256 + verifier) suitable for the SatCen D3 deliverable.

---

## Real validation numbers (xView3-SAR Mediterranean / Adriatic, 11 scenes)

Measured on `468 575 km²` of Sentinel-1 GRD, **1 997 ground-truth
vessels** (xView3 confidence ≥ MEDIUM, `is_vessel=True`), match
mode `center` ≤ 20 px (the official xView3-SAR scoring convention).

| Detector | mAP | Pd (recall) | FAR / km² | Precision | Predictions |
|---|---:|---:|---:|---:|---:|
| `cfar-default` (baseline) | 0.0104 | **0.4226** | 0.1157 | 0.0153 | 55 064 |
| `vesseltracker-sar-yolov8` | 0.0242 | 0.1432 | **0.0041** | **0.1305** | 2 191 |

CFAR catches 3× more vessels but emits 25× more detections; YOLO
filters port/glint clutter at the cost of recall. The
**production fusion** path of AIDRA combines both — the architectural
decision behind that combination is now backed by real measurements,
not just a hunch. Full per-scene tables and caveats live in the
[MODEL_CARDs](models/cards/).

---

## Constrained-hardware target

| Profile | CPU | RAM | TDP (W) | Simulates |
|---|---:|---:|---:|---|
| `ground` | 4.0 | 24 GB | 65 | Ground station baseline |
| `sat-high` | 2.0 | 4 GB | 10 | Xilinx Zynq UltraScale+ class |
| `sat-mid` | 1.0 | 2 GB | 5 | NXP LX2160A / Unibap iX5 class |
| `sat-low` | 0.5 | 1 GB | 2.5 | Raspberry Pi 4 class |
| `sat-extreme` | 0.25 | 512 MB | 1.5 | Cortex-M / RP2040 break-point |

Resource caps are enforced via `setrlimit` and CPU affinity (`sched_setaffinity` on Linux). `ResourceCollector` samples
RAM/CPU at 100 ms cadence and emits `latency_p95_ms` and
`energy_estimated_j` (`avg_cpu_fraction × tdp_watts × duration`).

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│           OCI ARM A1 Free Tier (4 OCPU, 24 GB)           │
│                                                          │
│  ┌──────────────────────────────┐    ┌────────────────┐  │
│  │ Python App (FastAPI)         │    │ Grafana :3000  │  │
│  │  :8000                       │    │                │  │
│  │  ┌─────────┐  ┌────────────┐ │    │ • Vessel map   │  │
│  │  │ API     │  │ Pipeline   │ │    │ • Pipeline KPI │  │
│  │  │ REST    │  │ engine     │ │    │ • Profile bench│  │
│  │  │         │  │            │ │    │ • Traceability │  │
│  │  │ /detec. │  │ ingest →   │ │    └────────────────┘  │
│  │  │ /stac   │  │ preprocess │ │    ┌────────────────┐  │
│  │  │ /ogc    │  │ detect     │ │────│ PostgreSQL +   │  │
│  │  │ /trace  │  │ persist    │ │    │   PostGIS      │  │
│  │  │ /bench  │  │ cleanup    │ │    │  :5432         │  │
│  │  └─────────┘  └────────────┘ │    └────────────────┘  │
│  │  ┌──────────────┐            │                         │
│  │  │ APScheduler  │            │    ┌─────────────┐     │
│  │  │ (cron+cue)   │            │    │ Prometheus  │     │
│  │  └──────────────┘            │    │  + Loki     │     │
│  └──────────────────────────────┘    └─────────────┘     │
└──────────────────────────────────────────────────────────┘
```

Stack: **100 % Python (≥ 3.11)** — FastAPI · PyTorch · ultralytics ·
rasterio · APScheduler · psutil · prometheus-client · asyncpg
· pydantic-settings.

---

## Quick start

```bash
git clone https://github.com/RadW2020/aidra-feasibility-study.git
cd aidra-feasibility-study
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env                          # edit Copernicus creds, etc.
docker compose up -d aidra-db                  # local PostGIS
psql "$DATABASE_URL" -f src/db/migrations/001_init.sql
psql "$DATABASE_URL" -f src/db/migrations/002_indexes.sql
psql "$DATABASE_URL" -f src/db/migrations/003_tipcue.sql
psql "$DATABASE_URL" -f src/db/migrations/004_traceability.sql
psql "$DATABASE_URL" -f src/db/migrations/005_thumbnails.sql
psql "$DATABASE_URL" -f src/db/migrations/006_resilience.sql
psql "$DATABASE_URL" -f src/db/migrations/007_normalize_compression.sql
psql "$DATABASE_URL" -f src/db/migrations/008_detection_quality.sql
./scripts/download-models.sh                   # pulls vesseltracker-sar-yolov8.pt
python -m src.main                             # FastAPI on :8000
```

Browse the OpenAPI docs at <http://localhost:8000/docs>.

### Run a scene end-to-end

```bash
curl -X POST http://localhost:8000/api/pipeline/trigger \
     -H 'Content-Type: application/json' \
     -d '{"zone": "gibraltar", "model": "vesseltracker-sar-yolov8", "profile": "ground"}'
```

### Pull GeoJSON for QGIS

```bash
curl 'http://localhost:8000/api/detections.geojson?min_confidence=0.5&on_land=false&cluster_anomaly=false' \
     -o detections.geojson
```

### Run validation against a labelled manifest

```bash
python -m scripts.run_validation \
    --manifest data/validation/your_manifest.json \
    --model vesseltracker-sar-yolov8 \
    --output reports/validation.json \
    --match-mode center --center-tolerance-px 20
```

A reproducible synthetic baseline ships in `scripts/build_synthetic_manifest.py`. The xView3-SAR Mediterranean
pipeline is documented step-by-step in `models/cards/cfar-default.MODEL_CARD.md` (D2 section).

---

## Repository layout

```
src/
  api/              FastAPI routers (detections, stac, ogc_features, ...)
  db/               PostgreSQL/PostGIS schema + asyncpg queries
  models/           Detector wrappers (CFAR, YOLO) + ModelManager + cards gate
  pipeline/         ingest → preprocess → detect → postprocess → bundle
  profiles/         Constraint profiles (CPU/RAM/TDP) + ResourceCollector
  observability/    Prometheus metrics (with run_id exemplars) + Loki logger
  orbital/          Tip&Cue, energy, downlink, latency, resilience
  tipcue/           Re-tasking evaluator + scheduler
  traceability/     SHA256 hasher + recorder + verifier + D3 bundler
models/
  *.pt              Active model weights (only with a MODEL_CARD)
  archived/         Weights retired from evaluation (provenance unknown)
  cards/            MODEL_CARD.md per model — AI Act gate I-AIA-1
scripts/
  filter_xview3_med.py        Filter xView3-SAR validation to Med subset
  build_xview3_manifest.py    xView3 CSV labels → AIDRA manifest schema
  validate_xview3_serial.py   Disk-tight serial driver for xView3 evaluation
  run_validation.py           Generic harness (mAP / Pd / FAR / PR-curve)
  build_synthetic_manifest.py Reproducible synthetic SAR baseline
  run_interpretability.py     Grad-CAM + CFAR heatmap renderer (D4)
  download-models.sh          Pull weights from declared sources
tests/
  test_pipeline/    Edge filter (I-SAR-2), preprocess, detection helpers
  test_traceability/ Hasher, bundler, e2e reproducibility
  test_models/      AI Act gate, CFAR
  test_profiles/    ResourceCollector + percentile + energy
  test_validation/  Validation harness
  test_api/         OGC Features, STAC search, detections, metrics
  test_invariants.py All declared invariants from CLAUDE.md §5
```

---

## Documentation index

| File | What | When to read |
|---|---|---|
| [`mvp_oci.md`](mvp_oci.md) | Implementation plan + formal **WBS / Gantt / per-WP load** + risk register links. | Executive overview |
| [`TECHNICAL_SPEC.md`](TECHNICAL_SPEC.md) | Full technical reference — interfaces, SQL, APIs, profiles, compression, Tip & Cue (4500+ lines). | Implementation reference |
| [`CLAUDE.md`](CLAUDE.md) | LLM operating contract for the repo — invariants (§5), DoD (§3), gates (§6). | Before touching code |
| [`AI_ACT_DECLARATION.md`](AI_ACT_DECLARATION.md) | EU AI Act 2024/1689 voluntary classification + Art. 14 governance. | D1 / D4 deliverables |
| [`D4_INTERPRETABILITY_ANNEX.md`](D4_INTERPRETABILITY_ANNEX.md) | Grad-CAM and CFAR heatmap samples on real production runs. | D4 deliverable |
| [`RISK_REGISTER.md`](RISK_REGISTER.md) | Live risk + mitigation log (Copernicus quota, OCI reclaim, dataset license, AI Act re-classification, drift, reproducibility, geographic bias, **R8 Terrain Correction formal scope exclusion**). | Risk reviews |
| [`EVALUATOR_GUIDE.md`](EVALUATOR_GUIDE.md) | One-page guide for a tender evaluator — what to read in which order. | Reviewer onboarding |
| [`analisis_completo.md`](analisis_completo.md) | Reference to the original SatCen tender (Spanish). | Tender clarifications |

---

## Quality gates

| Gate | Command | Status |
|---|---|---|
| Lint | `ruff check src/ tests/ scripts/` | ✅ All checks passed |
| Tests | `pytest -q` | ✅ 245 / 245 |
| Invariants | `pytest -k invariant -x` | ✅ Enforced (I-SAR-1..3, I-DET-2..3, I-MOD-4, I-TRACE-1..4, I-AIA-1) |
| Reproducibility | `pytest -k reproducibility -x` | ✅ Same input → same `output_hash` end-to-end |
| AI Act gate | `ModelManager` refuses any weight without a `MODEL_CARD.md` | ✅ Tested |

---

## What is **out of scope**

- **Range-Doppler Terrain Correction** is a formally documented
  scope exclusion (`R8` in `RISK_REGISTER.md`). The pipeline geocodes
  via affine GCPs, valid for open-sea AOIs (Gibraltar / Suez /
  Red Sea / English Channel). Not valid for high-relief coastline.
- **Re-training a custom model** — AIDRA evaluates pre-trained
  weights with full provenance, it does not produce a new SOTA
  detector.
- **In-orbit deployment** — TRL ≤ 6 by design. Validation in
  real space hardware is on the roadmap (Orion testbed,
  ESA Φsat-2 visiting researcher).
- **CE marking / regulatory deployment** — covered as a documented
  reclassification trigger in the AI Act declaration; not in scope
  for the POC.

---

## Datasets

| Dataset | Use | Status |
|---|---|---|
| **Copernicus Sentinel-1 GRD** | Operational input via Copernicus Data Space (free, EU-hosted) | Verified, 12 TB / month free quota |
| **xView3-SAR** | D2 formal validation (mAP / Pd / FAR) — Mediterranean subset | Verified, 11 Mediterranean / Adriatic validation scenes used |
| **HRSID** | Optional sanity check | Verified |
| **OpenSARShip** | Optional sanity check | Verified |

The labelled `xView3-SAR` files are **not** redistributed in this
repo (research-only EULA from Maxar / DIU). Download the labels via
`iuu.xview.us` and place them in `x-view-us-data/` — gitignored.

---

## Hardware-validation roadmap

| Priority | Platform | Value | Status |
|---|---|---|---|
| Immediate | [Orion CubeSat Testbed](https://github.com/omega-space-group/orion-cubesat-testbed) | Open-source flatsat (FPGA / GPU / neuromorphic) — benchmarks on real CubeSat hardware | Integration pending |
| Nice-to-have | [ESA Φsat-2 Visiting Researcher](https://cin.philab.esa.int/schemes/visiting-researchers-in-onboard-ai-for-sat-2-mission) | 6U CubeSat in orbit with on-board AI app platform — TRL 7 validation. Vessel detection is a declared use case. | Application pending |

---

## License

The **AIDRA source code** in this repository is licensed under the
[MIT License](LICENSE).

The repository is suitable for code review and research reproduction as-is.
The CFAR path is MIT-covered. The YOLOv8 path imports `ultralytics`; keep the
license note below in mind before deploying it as a public network service.

> ⚠️ **AGPL-3.0 runtime dependency** — AIDRA imports
> [`ultralytics`](https://github.com/ultralytics/ultralytics) (the
> reference YOLOv8 implementation) at runtime in
> `src/models/yolo.py` and the three modules under
> `src/models/compression/`. The `ultralytics` package is licensed
> under **AGPL-3.0-or-later**, with a separate Ultralytics
> Enterprise License available for commercial/closed-source use.
>
> The MIT license over the AIDRA source code does **not** override
> the obligations of AGPL-3.0 over the combined runtime. In
> particular, AGPL-3.0 §13 (*"Remote Network Interaction"*) applies
> to anyone who exposes this software as a public network service
> while keeping the YOLOv8 detection path active. Operators must
> choose one of:
>
> 1. **Comply with AGPL-3.0** — publish all source for the combined
>    work (including local modifications) under AGPL-3.0 to all
>    users of the network service.
> 2. **Drop the dependency** — replace `ultralytics` with an in-house
>    inference backend that loads the weights directly with
>    `torch.load` / ONNX Runtime. AIDRA is structured so this swap
>    is local to `src/models/yolo.py` and the compression modules.
> 3. **Buy an Ultralytics Enterprise License** for the deployment.
>
> This notice is informational, not legal advice.

Other third-party assets:

- **Sentinel-1 imagery** — Copernicus open license (free reuse,
  EU-hosted via [dataspace.copernicus.eu](https://dataspace.copernicus.eu)).
- **xView3-SAR labels and scenes** — research-only EULA from Maxar /
  Defense Innovation Unit. Not included in this repository; download
  it yourself from `iuu.xview.us` and place it under
  `x-view-us-data/` (gitignored).
- **YOLOv8 model weights** — distributed by Ultralytics under
  AGPL-3.0. Any locally-trained derivative weights inherit the
  AGPL-3.0 unless re-licensed via Ultralytics Enterprise.
