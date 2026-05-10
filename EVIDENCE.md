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
