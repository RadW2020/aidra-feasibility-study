# AIDRA Evidence Index

Produced deliverables for SatCen tender SATCEN/2026/OP/0003. Each entry
points to (a) the canonical artifact (server-side), (b) what is mirrored
into git for auditability without downloads, and (c) the verification
command.

---

## D3 — Evidence Bundle

| | |
|---|---|
| Bundle ID | `d3-20260506T080713Z` |
| Generated | 2026-05-06 08:07 UTC |
| Trigger | `POST /api/interpretability/run` (deployed AIDRA, image `c268a61`) |
| Manifest commit_sha (annotated) | `92b2515` *(see annotation bug below)* |
| Actual code commit | `c268a61eb189dd6a90767adbe35ea6eeddbfcad7` (env `SOURCE_COMMIT`) |
| Files in bundle | 22 243 |
| Bundle archive size | 72 MB (gzip) |
| Bundle root SHA256 | `68c22a684ab192d4a2cc7c505f76f1a575a0e8fb1465b3d72ee530efc2359bc9` |
| MANIFEST.json SHA256 | `43c11b3625b34ce6836fdfd7981b46283679ce9f5d42c641c862bb882990ed91` |
| Server path | `aidra.uliber.com:/data/interpretability/d3_bundles/d3-20260506T080713Z.tar.gz` (inside `aidra-app` container, mounted on host volume `aidra-interpretability`) |
| Local download path | `evidence_bundles/d3-20260506T080713Z.tar.gz` *(gitignored)* |

**Mirrored to git** (under `evidence_bundles/`):
- `d3-20260506T080713Z.MANIFEST.json` — full file-by-file SHA256 inventory.
- `d3-20260506T080713Z.MANIFEST.sha256` — root signature; one line, single auditor anchor.
- `d3-20260506T080713Z.settings.json` — Settings snapshot at bundle build time.
- `d3-20260506T080713Z.executions.csv` — flat dump of `execution_log` rows included in the bundle.

**Verify** (against the local tarball or extracted dir):
```bash
shasum -a 256 evidence_bundles/d3-20260506T080713Z.tar.gz
# expected: 68c22a684ab192d4a2cc7c505f76f1a575a0e8fb1465b3d72ee530efc2359bc9

mkdir -p /tmp/d3 && tar -xzf evidence_bundles/d3-20260506T080713Z.tar.gz -C /tmp/d3
.venv/bin/python -m src.traceability verify-bundle /tmp/d3/d3-20260506T080713Z
# expected: Result: PASS — 22243/22243 files OK, MANIFEST root OK
```

Last verification: **PASS** (22 243 files OK, 0 mismatches, 0 missing, 0 extras, settings_hash OK, MANIFEST root OK).

---

## D4 — Interpretability run (Grad-CAM + CFAR)

| | |
|---|---|
| Run ID | `173bbdb5-aaf2-489d-8e34-cec4243705d4_interp_bcfb083c` |
| Source execution | `173bbdb5-aaf2-489d-8e34-cec4243705d4` |
| Generated | 2026-05-06 08:06 UTC |
| Trigger | `POST /api/interpretability/run` (n_samples=20) |
| Model used for explanations | `vesseltracker-sar-yolov8` (FP32 PT baseline, loaded directly per the I-AIA-2 fix in `a8dbfda`) |
| Source-execution `model_hash` (annotated) | `ea0ee6dac…` *(INT8 ONNX — see annotation bug below)* |
| Grad-CAM success rate | **20 / 20** |
| CFAR success rate | **20 / 20** |
| Per-PNG SHA256 mismatches | 0 |
| Server path | `aidra.uliber.com:/data/interpretability/173bbdb5-aaf2-489d-8e34-cec4243705d4_interp_bcfb083c/` |
| Local download path | `interpretability_runs/interp_run.tar.gz` *(gitignored)* |

**Mirrored to git** (under `interpretability_runs/`):
- `manifest.json` — full per-sample SHA256 + commit/model anchors.
- `samples/000_*.png`, `samples/009_*.png`, `samples/019_*.png` — first / middle / last
  triplets (input SAR / Grad-CAM overlay / CFAR score map). Visual evidence
  of the explainability artifact without needing to extract the tarball.

**Verify**:
```bash
mkdir -p /tmp/d4 && tar -xzf interpretability_runs/interp_run.tar.gz -C /tmp/d4
.venv/bin/python -c "
import json, hashlib
from pathlib import Path
d = Path('/tmp/d4/173bbdb5-aaf2-489d-8e34-cec4243705d4_interp_bcfb083c')
m = json.load((d/'manifest.json').open())
mismatches = sum(
    1
    for s in m['samples']
    for kind in ('input', 'gradcam', 'cfar')
    if s.get(f'{kind}_png')
    and hashlib.sha256((d/s[f'{kind}_png']).read_bytes()).hexdigest() != s[f'{kind}_sha256']
)
print(f'PNG SHA256 mismatches: {mismatches}')
"
# expected: 0
```

---

## Known annotation bugs (do not affect data integrity)

Both deliverables verify cleanly. Two annotation-only issues worth noting
to the auditor; fixes tracked separately:

1. **`commit_sha` in MANIFEST is stale (`92b2515`).** The Coolify
   environment defines `AIDRA_COMMIT_SHA` as a hardcoded build-arg that
   was not refreshed on the latest deploys. The actually-running code is
   commit `c268a61` (per `SOURCE_COMMIT`, container image tag, and
   OpenAPI surface — both new endpoints `/api/interpretability/run` and
   `/api/traceability/bundle` are present). Mitigation: `get_commit_sha()`
   in `src/traceability/hasher.py` should prefer `SOURCE_COMMIT` over
   `AIDRA_COMMIT_SHA`, or read from a baked-in version file at build
   time.

2. **D4 manifest reports the source-execution `model_hash` (INT8 ONNX),
   not the FP32 PT actually used for Grad-CAM.** This is correct in spirit
   — the explanations are *for* an INT8 ONNX detector run — but the
   runtime artifact that produced the heatmaps is the architecturally-
   identical FP32 baseline (Grad-CAM needs autograd; ONNX has no gradient
   graph). Fix: extend the manifest schema to record both
   `execution_model_hash` (subject of the explanation) and
   `gradcam_model_hash` (provenance of the heatmap renderer).

---

## Reproduction (for the auditor)

The whole chain is reproducible from outside the server given API token.
Both endpoints are authenticated via `Authorization: Bearer
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
