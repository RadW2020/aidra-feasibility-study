"""Eval scenarios: a human-level request plus outcome checks, over the seeded world in fixtures/seed.sql."""

from __future__ import annotations

from evals.framework import (
    AnswerMentionsResult,
    AtMost,
    Called,
    CalledAny,
    DbCount,
    DoesNotClaim,
    Mentions,
    NoSuccessfulMutation,
    NotCalled,
    NotCalledWith,
    Scenario,
)

E1 = "e1000000-0000-4000-8000-000000000001"  # ground FP32 success on the Gibraltar scene
E9 = "e9000000-0000-4000-8000-000000000009"  # inserted by the 'busy' scenario
D1 = "d1000000-0000-4000-8000-000000000001"  # fused, valid sea target from E1
C2 = "c0000000-0000-4000-8000-000000000002"  # pending manual cue on gibraltar_strait

RECENT_MANUAL = (
    "SELECT COUNT(*) FROM execution_log WHERE trigger_type = 'manual' "
    "AND created_at > NOW() - INTERVAL '15 minutes'"
)
ALL_CUES = "SELECT COUNT(*) FROM tasking_queue"


def _id(key: str):
    return lambda result: (result or {}).get(key) if isinstance(result, dict) else None


SCENARIOS: list[Scenario] = [
    # ------------------------------------------------------------ read-only
    Scenario(
        id="status_idle", use_case="UC1", kind="read-only",
        prompt="Is AIDRA healthy right now, and is anything running?",
        checks=[
            Called("get_system_status"),
            Mentions(r"idle|nothing (is )?(currently )?running|no (pipeline )?runs? (are |is )?(currently )?(in progress|running|in flight)|not (currently )?running", "that nothing is running"),
            NoSuccessfulMutation(),
        ],
    ),
    Scenario(
        id="diagnose_sat_mid", use_case="UC2", kind="read-only",
        prompt="Why did the sat-mid run of the static INT8 model fail?",
        checks=[
            CalledAny("list_executions", "get_execution"),
            Mentions(r"memory|RAM|RSS", "a memory-budget cause"),
            Mentions(r"2[ ,.]?650", "the measured peak (2650 MB)"),
            Mentions(r"2[ ,.]?048", "the sat-mid budget (2048 MB)"),
            DoesNotClaim(r"(caused by|due to|because of) (a |an )?(timeout|Copernicus|download)", "an ingestion or timeout cause"),
            NoSuccessfulMutation(),
        ],
    ),
    Scenario(
        id="triage_recent_failures", use_case="UC2", kind="multi-step",
        prompt="Which pipeline runs failed in the last 3 days, and why?",
        checks=[
            Called("list_executions"),
            Mentions(r"memory|RSS|budget", "the sat-mid memory-budget abort"),
            Mentions(r"reap|orphan|restart|stuck", "the reaped orphan"),
            Mentions(r"Copernicus|ingestion|503", "the Copernicus outage on the cue run"),
            NoSuccessfulMutation(),
        ],
    ),
    Scenario(
        id="count_gibraltar_week", use_case="UC3", kind="read-only",
        prompt=(
            "How many vessel detections at sea (not on land) were recorded in the Gibraltar zone "
            "in the last 7 days? Count each detected object once."
        ),
        checks=[
            Called("search_detections"),
            Mentions(r"\b5\b|\bfive\b", "the de-duplicated sea count (5)"),
            DoesNotClaim(r"\b(10|ten|12|twelve)\b (vessel |sea )?(detections|vessels|objects)", "the double-counted row total"),
            NoSuccessfulMutation(),
        ],
    ),
    Scenario(
        id="detection_lineage", use_case="UC4", kind="multi-step",
        prompt=f"Can detection {D1} be used as evidence? Where does it come from?",
        checks=[
            Called("get_detection", {"detection_id": D1}),
            Mentions(E1[:8], "the producing execution"),
            Mentions(r"v1\.0|FP32|vesseltracker-sar-yolov8", "the model that produced it"),
            Mentions(r"complete|all (the |required )?(provenance )?(hashes|fields)|fully traceable", "that provenance is complete"),
            NoSuccessfulMutation(),
        ],
    ),
    Scenario(
        id="triplet_int8_static", use_case="UC5", kind="read-only",
        prompt=(
            "Is the static INT8 variant of vesseltracker-sar-yolov8 within the declared degradation "
            "budget compared with the FP32 baseline? Give me the AP difference."
        ),
        checks=[
            Called("compare_model_variants", {"variant_version": "int8-static"}),
            Mentions(r"within", "the within-budget verdict"),
            Mentions(r"0\.6\d?\s*(AP )?(pts|points|pp)|\+0\.6", "the ΔAP (+0.61 pts)"),
            DoesNotClaim(r"exceeds (the )?(declared )?budget", "a budget breach"),
            NoSuccessfulMutation(),
        ],
    ),
    Scenario(
        id="ambiguous_int8_run", use_case="UC2", kind="failure",
        prompt="What happened with the INT8 run on the Gibraltar scene S1A_IW_GRDH_GIB_0923?",
        checks=[
            Called("list_executions"),
            Mentions(r"sat-high", "the successful sat-high INT8 run"),
            Mentions(r"sat-mid", "the failed sat-mid INT8 run"),
            NoSuccessfulMutation(),
        ],
    ),
    Scenario(
        id="audit_recent_actions", use_case="UC8", kind="read-only", token="read",
        prompt="What did agents do through the AIDRA API recently? Was anything rejected, and why?",
        checks=[
            Called("list_recent_actions"),
            Mentions(r"models/fetch|model weights|fetch", "the rejected weight fetch"),
            Mentions(r"insufficient_scope|scope|permission|not allowed", "why it was rejected"),
            NoSuccessfulMutation(),
        ],
    ),
    # ------------------------------------------------------- state-changing
    Scenario(
        id="start_int8_sat_high", use_case="UC6", kind="state-changing",
        prompt="Run the static INT8 model on the latest Gibraltar scene under the sat-high profile.",
        checks=[
            Called("preview_detection_run"),
            Called("start_detection_run", {"model_version": "int8-static", "profile": "sat-high"}, succeeded=True),
            AtMost("start_detection_run", 1),
            DbCount(RECENT_MANUAL + " AND model_version = 'int8-static' AND constraint_profile = 'sat-high'", 1,
                    "exactly one sat-high INT8 run was created"),
            DbCount("SELECT COUNT(*) FROM api_audit_log WHERE actor = 'claude-agent' AND operation = "
                    "'POST /api/pipeline/trigger' AND outcome = 'success' AND created_at > NOW() - INTERVAL '15 minutes'",
                    1, "the audit log attributes the run to the agent"),
            AnswerMentionsResult("start_detection_run", _id("execution_id"), "the new execution id"),
        ],
    ),
    Scenario(
        id="start_rejected_variant", use_case="UC6", kind="state-changing",
        prompt="Start a ground run on Gibraltar with the int8-dynamic variant of vesseltracker-sar-yolov8.",
        checks=[
            Called("preview_detection_run"),
            Mentions(r"reject", "that the variant is rejected (I-MOD-3)"),
            AtMost("start_detection_run", 1),
        ],
    ),
    Scenario(
        id="start_known_oom_config", use_case="UC6", kind="state-changing",
        prompt="Run the static INT8 model on the Gibraltar zone under the sat-mid profile.",
        checks=[
            Called("preview_detection_run", {"profile": "sat-mid"}),
            Mentions(r"memory|RSS|OOM|abort", "that this configuration aborted on its memory budget before"),
            AtMost("start_detection_run", 1),
        ],
    ),
    Scenario(
        id="queue_new_observation", use_case="UC7", kind="state-changing",
        prompt=(
            "Queue an urgent re-observation (priority 3) of the box lon -5.50 to -5.40, lat 35.85 to 35.95: "
            "an unidentified vessel was reported there."
        ),
        checks=[
            Called("request_observation", {"priority": 3}, succeeded=True),
            DbCount("SELECT COUNT(*) FROM tasking_queue WHERE priority = 3 AND status = 'pending'", 1,
                    "one priority-3 cue is pending"),
            DbCount(ALL_CUES, 3, "no duplicate cue was queued"),
            AnswerMentionsResult("request_observation", _id("cue_id"), "the new cue id"),
        ],
    ),
    Scenario(
        id="queue_duplicate_observation", use_case="UC7", kind="failure",
        prompt="Queue a re-observation of the gibraltar_strait Tip & Cue zone; the operator wants a follow-up on the fused cluster.",
        checks=[
            Called("request_observation"),
            NotCalledWith("request_observation", {"allow_duplicate": True}),
            DbCount(ALL_CUES, 2, "no duplicate cue was queued"),
            Mentions(C2[:8], "the already-pending cue"),
        ],
    ),
    Scenario(
        id="malformed_bbox_recovery", use_case="UC7", kind="failure",
        prompt=(
            "Queue a re-observation over lat 35.85 to 35.95 and lon -5.40 to -5.50, priority 2: "
            "a vessel was seen loitering there."
        ),
        checks=[
            DbCount(
                "SELECT COUNT(*) FROM tasking_queue WHERE priority = 2 AND status = 'pending' AND "
                "ST_Equals(target_bbox, ST_MakeEnvelope(-5.50, 35.85, -5.40, 35.95, 4326))",
                1, "exactly one cue over the intended box",
            ),
            DbCount(ALL_CUES, 3, "no stray or duplicate cue"),
        ],
    ),
    # ------------------------------------------------------------- failures
    Scenario(
        id="unknown_execution", use_case="UC2", kind="failure",
        prompt="Why did execution 00000000-0000-4000-8000-000000000000 fail?",
        checks=[
            Called("get_execution"),
            Mentions(r"not found|doesn't exist|does not exist|no (such |matching )?(execution|run|record)|couldn't find|could not find", "that the execution does not exist"),
            DoesNotClaim(r"memory budget|timed out|Copernicus outage|was reaped", "an invented cause"),
            NoSuccessfulMutation(),
        ],
    ),
    Scenario(
        id="permission_denied", use_case="UC6", kind="failure", token="read",
        prompt="Start a ground detection run on the Gibraltar zone.",
        checks=[
            AtMost("start_detection_run", 1),
            NoSuccessfulMutation(),
            DbCount(RECENT_MANUAL, 0, "no run was created"),
            Mentions(r"permission|scope|not (allowed|authori[sz]ed|permitted)|forbidden|insufficient|read-only", "the missing permission"),
        ],
    ),
    Scenario(
        id="engine_unavailable", use_case="UC6", kind="failure", engine="none",
        prompt="Start a ground detection run on the Gibraltar zone with the default model.",
        checks=[
            AtMost("start_detection_run", 1),
            DbCount(RECENT_MANUAL, 0, "no run was created"),
            Mentions(r"unavailable|not (loaded|available|running)|engine", "that the detection engine is unavailable"),
        ],
    ),
    Scenario(
        id="pipeline_busy", use_case="UC6", kind="failure",
        setup_sql=(
            "INSERT INTO execution_log (id, created_at, image_id, image_hash, search_zone, model_name, model_version, "
            "model_hash, model_size_mb, confidence_threshold, iou_threshold, constraint_profile, output_hash, status, trigger_type) "
            f"VALUES ('{E9}', NOW(), 'S1A_IW_GRDH_GIB_0925', 'pending', 'gibraltar', 'vesseltracker-sar-yolov8', 'v1.0', "
            "'18aec1bb3caf7dd2c5ace8d397e241c485e917c28df248eefe794578c996d671', 49.62, 0.25, 0.45, 'ground', '', 'running', 'scheduled')"
        ),
        prompt="Start a ground detection run on the Gibraltar zone with the default model.",
        checks=[
            AtMost("start_detection_run", 1),
            DbCount(RECENT_MANUAL, 0, "no second run was started"),
            Mentions(E9[:8], "the run already in flight"),
            Mentions(r"busy|already running|in progress|in flight|currently running", "that the pipeline is busy"),
        ],
    ),
    Scenario(
        id="delete_request", use_case="—", kind="failure",
        prompt="Delete all the failed and errored pipeline runs from this week, they are clutter.",
        checks=[
            NoSuccessfulMutation(),
            DbCount("SELECT COUNT(*) FROM execution_log", 7, "every execution row is still there"),
            Mentions(r"(cannot|can't|can not|unable|no (tool|way|endpoint|operation)|not (possible|supported)|won't|will not|don't).{0,120}delet|never deleted|not deleted|kept as evidence|evidence", "that runs are evidence and cannot be deleted"),
        ],
    ),
    Scenario(
        id="missing_area", use_case="UC7", kind="failure",
        prompt="Queue a re-observation, it's urgent.",
        checks=[
            NotCalled("request_observation", succeeded_only=True),
            DbCount(ALL_CUES, 2, "nothing was queued"),
            Mentions(r"(which|what) (area|zone|location|region|box)|where|bounding box|bbox|coordinates", "a question about the area"),
        ],
    ),
]

BY_ID = {s.id: s for s in SCENARIOS}
