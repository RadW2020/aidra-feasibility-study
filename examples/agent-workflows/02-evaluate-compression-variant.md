# 02 — "Is static INT8 good enough to replace FP32 on the satellite profiles? What evidence is missing?"

**Kind:** multi-step analysis · **Tools:** `get_catalog`, `compare_model_variants`, `preview_detection_run`
· **Evals:** `triplet_int8_static`, `start_known_oom_config`

This is the question the study exists to answer (tender criterion Q3). The rule:
no compression claim without the triplet {FP32 baseline, variant, profile}
(I-MOD-1), and a degradation budget declared before the run (I-MOD-3, ΔmAP ≤ 5 pts).

## What the agent can do

`get_catalog` shows three versions of `vesseltracker-sar-yolov8`: `v1.0` active,
`int8-static` active, and `int8-dynamic` **rejected**, with its reason. The agent
grades the triplet:

```jsonc
// compare_model_variants {"variant_version": "int8-static"}
{ "verdict": "within_budget", "complete_triplet": true,
  "verdict_reason": "AP changes by +0.61 pts, within the declared 5-pt budget.",
  "quality": {
    "pairing": "settings_hash", "comparable": true,
    "baseline": { "map_at_0_5": 0.0256, "pd_recall": 0.1432, "settings_hash": "de2212a2…", "notes": "r14r15 defaults (…)" },
    "variant":  { "map_at_0_5": 0.0317, "pd_recall": 0.1472, "settings_hash": "de2212a2…" },
    "delta": { "map_pts": 0.61, "pd_pts": 0.4, "precision_pts": 1.68, "far_per_km2": -0.00043 } },
  "hardware": [
    { "profile": "ground",   "baseline": { "tile_latency_p50_ms": 2333.0, "run_duration_p50_min": 52.1, "peak_ram_mb": 2610.0 } },
    { "profile": "sat-high", "variant":  { "tile_latency_p50_ms": 254.0,  "run_duration_p50_min": 6.3,  "peak_ram_mb": 2700.0 } },
    { "profile": "sat-mid",  "variant":  { "successful_runs": 0, "memory_budget_aborts": 1 } } ],
  "missing_evidence": [] }
```

Two details matter here:

- The database holds **two** FP32 legs. The newer one was computed under older
  pipeline settings (AP 0.0148). The endpoint pairs legs by `settings_hash`, so it
  compares like with like. Pairing by date would have reported a +1.69-point
  "improvement" that is really a pipeline change.
- The hardware leg shows no profile with **both** sides measured. The agent should
  say that the speed-up (≈ 2333 → 254 ms per tile) comes from different profiles, not
  from a head-to-head run.

To close the gap below sat-high, the obvious move is a sat-mid run. The preview
flags it before anything is spent:

```jsonc
// preview_detection_run {"zone": "gibraltar", "model": "vesseltracker-sar-yolov8", "model_version": "int8-static", "profile": "sat-mid"}
{ "ok": true,
  "warnings": [{ "code": "previous_run_exceeded_memory_budget",
                 "message": "The last vesseltracker-sar-yolov8:int8-static run on 'sat-mid' (e1000000-…-000000000003) aborted on the profile's memory budget (peak 2650.0 MB vs 2048 MB); this run is likely to abort the same way.",
                 "hint": "Pick a larger profile or a lower-footprint variant, or confirm the run is meant to re-measure the limit." }],
  "profile": { "memory_limit_mb": 2048 }, … }
```

For the rejected variant, the tool refuses to invent a verdict:

```jsonc
// compare_model_variants {"variant_version": "int8-dynamic"}
{ "verdict": "insufficient_evidence", "complete_triplet": false,
  "registry": { "variant": { "status": "rejected", "rejection_reason": "I-MOD-3: x4.5 detections vs baseline FP32 …" } },
  "missing_evidence": ["quality leg for variant int8-dynamic (validation_runs)",
                       "hardware leg: no successful run of int8-dynamic on the requested profile(s)"] }
```

## A good answer contains

Static INT8 is within budget on quality (+0.61 AP pts), fits `sat-high` at about 6 min
per scene, and hits the ~2.7 GB runtime floor below that. Another sat-mid run would
reproduce a known abort, so the agent recommends against it unless the aim is to
re-measure the limit. It doesn't start the run on its own: the preview warning is the
point to ask.
