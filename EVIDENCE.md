# AIDRA Evidence Index

Produced deliverables for SatCen tender SATCEN/2026/OP/0003. Each entry
points to (a) the canonical artifact (server-side), (b) what is mirrored
into git for auditability without downloads, and (c) the verification
command.

> **Run history.** First production run on 2026-05-06 08:07Z surfaced
> two annotation bugs in the manifests (stale `commit_sha` env var and
> conflated `model_hash`). Fix landed in `ed16ab7` and the run below
> (08:31Z) confirms both are resolved end-to-end. Old run is preserved
> in git history (commit `9832060`) for diff-against-fix auditability.

---

## D3 — Evidence Bundle

| | |
|---|---|
| Bundle ID | `d3-20260506T083120Z` |
| Generated | 2026-05-06 08:31 UTC |
| Trigger | `POST /api/traceability/bundle` (deployed AIDRA, image `ed16ab7`) |
| Manifest `commit_sha` | `ed16ab780a68b002b432214a98460873c09a3aab` ✅ matches `SOURCE_COMMIT` and the running container image tag |
| Counts | 21 executions, 50 985 detections, 22 236 thumbnails, 2 model cards |
| Bundle archive size | 72 MB (gzip) |
| Bundle root SHA256 | `6677d1a08d2ae89939d67007fec7048b4a62359c797e2f2fbc2a572750542f47` |
| Server path | `aidra.uliber.com:/data/interpretability/d3_bundles/d3-20260506T083120Z.tar.gz` (Docker volume `aidra-interpretability` on host) |
| Local download path | `evidence_bundles/d3-20260506T083120Z.tar.gz` *(gitignored)* |

**Mirrored to git** (under `evidence_bundles/`):
- `d3-20260506T083120Z.MANIFEST.json` — full file-by-file SHA256 inventory.
- `d3-20260506T083120Z.MANIFEST.sha256` — root signature; one line, single auditor anchor.
- `d3-20260506T083120Z.settings.json` — Settings snapshot at bundle build time.
- `d3-20260506T083120Z.executions.csv` — flat dump of `execution_log` rows included in the bundle.

**Verify** (against the local tarball or extracted dir):
```bash
shasum -a 256 evidence_bundles/d3-20260506T083120Z.tar.gz
# expected: 6677d1a08d2ae89939d67007fec7048b4a62359c797e2f2fbc2a572750542f47

mkdir -p /tmp/d3 && tar -xzf evidence_bundles/d3-20260506T083120Z.tar.gz -C /tmp/d3
.venv/bin/python -m src.traceability verify-bundle /tmp/d3/d3-20260506T083120Z
# expected: Result: PASS — 22243/22243 files OK, MANIFEST root OK
```

Last verification: **PASS** (22 243 files OK, 0 mismatches, 0 missing, 0 extras, settings_hash OK, MANIFEST root OK).

---

## D4 — Interpretability run (Grad-CAM + CFAR)

| | |
|---|---|
| Run ID | `173bbdb5-aaf2-489d-8e34-cec4243705d4_interp_227b8434` |
| Source execution | `173bbdb5-aaf2-489d-8e34-cec4243705d4` |
| Generated | 2026-05-06 08:31 UTC |
| Trigger | `POST /api/interpretability/run` (n_samples=20) |
| Manifest `commit_sha` | `ed16ab780a68b002b432214a98460873c09a3aab` ✅ |
| **Subject of explanation** (`execution_model_*`) | name `vesseltracker-sar-yolov8`, hash `ea0ee6da…` (= INT8 ONNX hash recorded in `execution_log`) |
| **Renderer of heatmap** (`gradcam_model_*`) | name `vesseltracker-sar-yolov8`, hash `18aec1bb…` (= FP32 PT baseline; only variant exposing autograd) |
| Two hashes differ | ✅ confirmed in manifest |
| Grad-CAM success rate | **20 / 20** |
| CFAR success rate | **20 / 20** |
| Per-PNG SHA256 mismatches | 0 |
| Server path | `aidra.uliber.com:/data/interpretability/173bbdb5-aaf2-489d-8e34-cec4243705d4_interp_227b8434/` |
| Local download path | `interpretability_runs/interp_run_v2.tar.gz` *(gitignored)* |

**Mirrored to git** (under `interpretability_runs/`):
- `manifest.json` — full per-sample SHA256 + commit + dual model anchors.
- `samples/000_*.png`, `samples/009_*.png`, `samples/019_*.png` — first / middle / last
  triplets (input SAR / Grad-CAM overlay / CFAR score map). Visual evidence
  of the explainability artifact without needing to extract the tarball.

**Verify**:
```bash
mkdir -p /tmp/d4 && tar -xzf interpretability_runs/interp_run_v2.tar.gz -C /tmp/d4
.venv/bin/python -c "
import json, hashlib
from pathlib import Path
d = Path('/tmp/d4/173bbdb5-aaf2-489d-8e34-cec4243705d4_interp_227b8434')
m = json.load((d/'manifest.json').open())
mismatches = sum(
    1
    for s in m['samples']
    for kind in ('input', 'gradcam', 'cfar')
    if s.get(f'{kind}_png')
    and hashlib.sha256((d/s[f'{kind}_png']).read_bytes()).hexdigest() != s[f'{kind}_sha256']
)
print(f'PNG SHA256 mismatches: {mismatches}')
print(f'execution_model_hash: {m[\"execution_model_hash\"][:16]}...')
print(f'gradcam_model_hash:   {m[\"gradcam_model_hash\"][:16]}...')
print(f'commit_sha:           {m[\"commit_sha\"]}')
"
# expected: 0 mismatches, two distinct model hashes, commit_sha = ed16ab78...
```

---

## Determinism evidence — `output_hash` profile-invariance (2026-05-08)

External audit on 2026-05-08 found that `output_hash` differed across
every re-run of the same scene (12 distinct hashes over 12 runs of the
same image+model+profile). Root cause: `Detection.id =
default_factory=uuid.uuid4` and the per-execution `thumbnail_path`
were both serialised into the result hash. Fix in commit `8214b44`
strips the non-content fields from the hash input.

The first batch run after the fix targeted Sentinel-1 product
`S1D_IW_GRDH_1SDV_20260505T062642_20260505T062707_002645_0046BC_4AA4`
(image_id `76f82d1c-…`) across the four profiles that completed
(`sat-extreme` exceeded the 60 min reaper threshold and was marked
failed — see R9 in the risk register).

**Correction (2026-05-10 external review):** these four rows were
originally described as FP32, but production SQL shows they were
`compression_technique='dynamic_int8'` and
`model_name='vesseltracker-sar-yolov8-int8-dynamic'`. They demonstrate
that the result hash no longer includes per-run UUID/thumbnail fields for
that particular INT8 batch, but they **do not close FP32 bitwise
reproducibility**.

| profile | execution_id (8) | num_detections | output_hash |
|---|---|---|---|
| ground | `dc837f54` | 1092 | `2c62f00608a38147…` |
| sat-high | `87ca8836` | 1092 | `2c62f00608a38147…` |
| sat-mid | `19f944f9` | 1092 | `2c62f00608a38147…` |
| sat-low | `258384b1` | 1092 | `2c62f00608a38147…` |

Single shared hash across four constraint profiles is useful regression
evidence for the hash canonicalisation fix, but **R6 remains pending for
an explicit FP32 rerun** after model-version pinning. Dynamic INT8 remains
tracked separately under R12 because other repeated INT8 groups still
produce distinct output hashes.

**Verify** (requires admin Grafana credentials or DB access):
```sql
SELECT constraint_profile, num_detections,
       model_name, compression_technique,
       LEFT(output_hash, 16) AS output_hash_prefix
FROM execution_log
WHERE id IN (
    'dc837f54-b908-4e82-92b7-7958b62e3faf',  -- ground
    '87ca8836-6d45-44ea-8782-1ebe35ef0cb9',  -- sat-high
    '19f944f9-c244-4ee4-9963-e61cae3a4748',  -- sat-mid
    '258384b1-90b9-47dd-97f5-e59b1d3b07b7'   -- sat-low
)
ORDER BY constraint_profile;
-- observed 2026-05-10: 4 rows, model_name='vesseltracker-sar-yolov8-int8-dynamic',
-- compression_technique='dynamic_int8', num_detections=1092 each,
-- output_hash_prefix='2c62f00608a38147'
```

---

## D4 (GT-anchored) — Grad-CAM / CFAR fidelity on xView3 (2026-08-29)

| | |
|---|---|
| Run | `d4_xview3_264ed833a13b7f2av_aidra_20260829T002845Z` — `reports/interpretability/xview3_264ed833a13b7f2av_baseline_p3_targeted/` |
| Sample | 20 chips (320 px = Grad-CAM input size, no resampling) from scene `264ed833a13b7f2av`, `aidra` set of the 2026-08-28 baseline: 5 × TP high-conf, TP low-conf, FP, FN (seed 42); repeated on the R14/R15 pipeline (`..._r14r15_p3_targeted/`) |
| Finding | Legacy Grad-CAM (P5 `model.model.21`, global target): pointing game 0/20, ≤ 0.7 % heat in box. New default (P3 `model.model.15`, detection-targeted): TP-high 3/5 (57 % heat in box), FP 2/5, TP-low 1/5, FN 1/5. CFAR score map: TP-high 4/5, FN 4/5 (4 % heat) |
| Provenance | `commit_sha` `b63188567554…`, renderer `vesseltracker-sar-yolov8` `18aec1bb…`, subject model hash from the validation report, CFAR 8/20, per-PNG SHA256 in `manifest.json` |
| Comparison runs | `..._baseline_p5_global/` (P5, global) and `..._baseline_p3_global/` (P3, global): `manifest.json` + `summary.md` |
| Production D4 | still the legacy run `173bbdb5…_interp_227b8434` above — regenerate after deploy (R16) |

**Verify**: `python -m scripts.run_interpretability_xview3 --dump reports/predictions/xview3_adriatic_full_vessels/264ed833a13b7f2av.json --tar data/xview3/scenes/264ed833a13b7f2av.tar.gz --report reports/validation_xview3_adriatic_full_vessels_aidra.json --set aidra --n-per-stratum 5 --chip 320 --seed 42 --out /tmp/d4check` → same `fidelity_summary` as the committed manifest.

---

## D2 — Detection quality against xView3-SAR ground truth (2026-08-28)

| | |
|---|---|
| Reports | `reports/validation_xview3_adriatic_full_vessels_{cfar,yolo,aidra,fused_only}.json` (+`.md`), `reports/validation_xview3_adriatic_detector_{yolo,cfar}_vessels.json` |
| Ground truth | xView3-SAR `validation.csv`, 11 Adriatic scenes, `is_vessel=True`, confidence HIGH+MEDIUM → 1 997 vessels, 468 575 km² |
| Settings | `confidence_threshold` 0.25, tiles 640/64, `edge_buffer_px` 32, `fusion_iou_threshold` 0.3 — all from `Settings` (I-DET-4), recorded in each report's `params` |
| Provenance (each report) | `commit_sha` `8ce9ca2e0f…`, `model_hash` `18aec1bb…` (SHA256 of `vesseltracker-sar-yolov8.pt`), `settings_hash`, seed 42, per-scene tar SHA256, library versions, steps exercised / not exercised |
| Steps not exercised | orbit correction, radiometric calibration, terrain correction — xView3 rasters are already SNAP-processed (see `provenance.steps_not_exercised`) |
| Reproducibility | Two independent full-pipeline runs (16:14Z and 18:29Z, different GT filters) produced **identical per-scene prediction counts** (18 257 predictions before clipping). Archived run: `reports/archived/validation_xview3_adriatic_full_allobjects_noclip_*.json` |
| Prediction dumps | `reports/predictions/xview3_adriatic_full_vessels/<scene_id>.json` — every prediction set + GT with `on_land`, for offline analysis and D4 sampling |
| Analysis | `reports/analysis_xview3_adriatic_full_vessels.md` — box geometry, YOLO↔CFAR co-location (42 %) vs IoU ≥ 0.3 (0.07 %), GT Venn split |
| Wall time | 4 365 s for 11 scenes on the workstation (`provenance.wall_time_s`) |

Headline (AIDRA production output, CFAR ∪ YOLO): Pd 0.356, FAR
0.0080/km², precision 0.159, AP 0.103, F1 0.220; sea-only Pd 0.448,
FAR 0.0067. Fused detections: 0 (R14). Full table in README.

**Verify**:
```bash
# provenance anchors present and consistent
python - <<'EOF'
import json, hashlib
r = json.load(open("reports/validation_xview3_adriatic_full_vessels_aidra.json"))
p = r["provenance"]
print("commit", p["commit_sha"], "| steps", len(p["steps"]), "| scenes hashed", len(p["image_hashes"]))
print("model_hash matches .pt:", hashlib.sha256(open("models/vesseltracker-sar-yolov8.pt","rb").read()).hexdigest() == p["model_hash"])
EOF
# regenerate (≈75 min, needs data/xview3/scenes/*.tar.gz and x-view-us-data/validation.csv)
python -m scripts.validate_xview3_serial --xview-dir x-view-us-data --tar-dir data/xview3/scenes \
  --tmp-dir data/xview3/scratch --models-dir models --seed 42 --vessels-only \
  --pipeline-path full --model all --output /tmp/d2_check.json
# expected: num_predictions 4460 (aidra), 3818 (cfar), 1442 (yolo), 0 (fused_only)
```

Import into production (after the migration 018 deploy; requires the API token):
```bash
python - <<'EOF' | curl -sS -X POST https://<deployed-aidra>/api/validation/import \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d @-
import json
r = json.load(open("reports/validation_xview3_adriatic_full_vessels_aidra.json"))
body = {k: r[k] for k in ("model_name","iou_threshold","confidence_threshold","num_scenes","num_ground_truth",
        "num_predictions","true_positives","false_positives","false_negatives","total_area_km2","pr_curve",
        "match_mode","center_tolerance_px","dataset","dataset_split","params","provenance")}
body.update(model_version="v1.0", model_hash=r["provenance"]["model_hash"], commit_sha=r["provenance"]["commit_sha"],
            pipeline_path="full", notes="xView3 Adriatic, vessels-only, full pipeline, 2026-08-28")
print(json.dumps(body))
EOF
```

---

## Reproduction (for the auditor)

The whole chain is reproducible from outside the server given an API
token. Both endpoints require `Authorization: Bearer
${AIDRA_API_TOKEN}`:

```bash
# D3
curl -X POST https://<deployed-aidra>/api/traceability/bundle \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"out_dir": "/data/interpretability/d3_bundles", "archive": true}'

# D4
curl -X POST https://<deployed-aidra>/api/interpretability/run \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"n_samples": 20, "model": "vesseltracker-sar-yolov8"}'
```

Both share their orchestration with the CLI scripts (`scripts/build_d3_bundle.py`,
`scripts/run_interpretability.py`); single source of truth.
