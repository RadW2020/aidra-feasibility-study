"""
CLI de trazabilidad AIDRA.

Subcomandos:
  verify         — verifica trazabilidad de un execution_id (DB).
  verify-bundle  — re-verifica un bundle D3 offline contra MANIFEST.
  bundle         — empaqueta el bundle D3.

Usage:
    python -m src.traceability bundle --out /tmp/d3 --model yolov8n-vessel
    python -m src.traceability bundle --out /tmp/d3 --zone gibraltar --no-archive
    python -m src.traceability verify <execution_id>
    python -m src.traceability verify-bundle /tmp/d3/d3-20260425T192447Z
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from uuid import UUID


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


async def _cmd_bundle(args: argparse.Namespace) -> int:
    from src.config import Settings
    from src.db.connection import db
    from src.traceability.bundler import EvidenceBundler

    settings = Settings()
    await db.connect(settings)
    try:
        bundler = EvidenceBundler(db=db, settings=settings)
        out_path = await bundler.build(
            out_dir=Path(args.out),
            date_from=_parse_iso(args.date_from),
            date_to=_parse_iso(args.date_to),
            zone=args.zone,
            model_name=args.model,
            constraint_profile=args.profile,
            archive=not args.no_archive,
        )
    finally:
        await db.disconnect()

    sys.stdout.write(str(out_path) + "\n")
    return 0


async def _cmd_verify(args: argparse.Namespace) -> int:
    # Lazy imports to avoid cost when only bundling.
    from src.config import Settings
    from src.db.connection import db
    from src.traceability.recorder import ExecutionRecorder

    settings = Settings()
    await db.connect(settings)
    try:
        recorder = ExecutionRecorder(db=db)
        record = await recorder.get(UUID(args.execution_id))
        if record is None:
            sys.stderr.write(
                f"Execution {args.execution_id} not found\n"
            )
            return 2
        # Print key traceability fields without requiring a re-run
        sys.stdout.write(
            f"id={record.id}\n"
            f"status={record.status}\n"
            f"image_hash={record.image_hash}\n"
            f"model_hash={record.model_hash}\n"
            f"output_hash={record.output_hash}\n"
            f"input_params_hash={record.input_params_hash}\n"
            f"commit_sha={record.commit_sha}\n"
        )
        return 0
    finally:
        await db.disconnect()


def _cmd_verify_bundle(args: argparse.Namespace) -> int:
    """Re-verify a D3 bundle offline against its MANIFEST.json."""
    from src.traceability.hasher import compute_sha256

    bundle = Path(args.bundle)
    if not bundle.exists():
        sys.stderr.write(f"Bundle not found: {bundle}\n")
        return 2

    manifest_path = bundle / "MANIFEST.json"
    if not manifest_path.exists():
        sys.stderr.write(f"MANIFEST.json missing under {bundle}\n")
        return 2

    import hashlib
    import json as _json

    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    expected_settings_hash = manifest.get("settings_hash")

    sys.stdout.write(f"Bundle: {bundle.name}\n")
    sys.stdout.write(f"Manifest commit_sha: {manifest.get('commit_sha')}\n")
    sys.stdout.write(
        f"Files declared in MANIFEST.json: {len(files)}\n\n"
    )

    failed: list[tuple[str, str, str]] = []
    missing: list[str] = []
    extras: list[str] = []
    on_disk: set[str] = set()

    for f in sorted(bundle.rglob("*")):
        if f.is_file() and f.name not in {"MANIFEST.json", "MANIFEST.sha256"}:
            on_disk.add(str(f.relative_to(bundle)))

    declared = set(files.keys())
    extras = sorted(on_disk - declared)
    missing = sorted(declared - on_disk)

    for rel, expected_sha in files.items():
        path = bundle / rel
        if not path.exists():
            continue
        actual_sha = compute_sha256(path)
        if actual_sha != expected_sha:
            failed.append((rel, expected_sha, actual_sha))

    # settings_hash double-check
    settings_path = bundle / "settings.json"
    settings_ok = True
    if settings_path.exists() and expected_settings_hash:
        actual = hashlib.sha256(
            settings_path.read_bytes()
        ).hexdigest()
        if actual != expected_settings_hash:
            settings_ok = False

    # MANIFEST.sha256 double-check (optional)
    manifest_root_ok = None
    root_sha_path = bundle / "MANIFEST.sha256"
    if root_sha_path.exists():
        expected = root_sha_path.read_text().split()[0]
        actual = compute_sha256(manifest_path)
        manifest_root_ok = expected == actual

    sys.stdout.write(
        f"Files OK:        {len(files) - len(failed) - len(missing)}\n"
        f"Files MISMATCH:  {len(failed)}\n"
        f"Files MISSING:   {len(missing)}\n"
        f"Files EXTRA:     {len(extras)}\n"
        f"settings_hash:   {'OK' if settings_ok else 'MISMATCH'}\n"
        f"MANIFEST root:   {'OK' if manifest_root_ok else ('MISMATCH' if manifest_root_ok is False else 'not signed')}\n"
    )
    for rel, e, a in failed[:10]:
        sys.stdout.write(f"  MISMATCH {rel}: expected {e[:16]}... got {a[:16]}...\n")
    for rel in missing[:10]:
        sys.stdout.write(f"  MISSING  {rel}\n")
    for rel in extras[:10]:
        sys.stdout.write(f"  EXTRA    {rel}\n")

    overall_ok = (
        not failed and not missing and settings_ok
        and (manifest_root_ok is None or manifest_root_ok)
    )
    sys.stdout.write(f"\nResult: {'PASS' if overall_ok else 'FAIL'}\n")
    return 0 if overall_ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="python -m src.traceability")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bundle = sub.add_parser("bundle", help="Build the D3 evidence bundle")
    p_bundle.add_argument("--out", required=True, help="Output directory")
    p_bundle.add_argument("--date-from", dest="date_from")
    p_bundle.add_argument("--date-to", dest="date_to")
    p_bundle.add_argument("--zone")
    p_bundle.add_argument("--model")
    p_bundle.add_argument("--profile")
    p_bundle.add_argument("--no-archive", action="store_true")

    p_verify = sub.add_parser("verify", help="Inspect traceability of a run")
    p_verify.add_argument("execution_id")

    p_vbundle = sub.add_parser(
        "verify-bundle", help="Re-verify a D3 bundle offline against its MANIFEST.json"
    )
    p_vbundle.add_argument("bundle", help="Path to bundle dir or bundle.tar.gz")

    args = parser.parse_args(argv)

    if args.cmd == "bundle":
        return asyncio.run(_cmd_bundle(args))
    if args.cmd == "verify":
        return asyncio.run(_cmd_verify(args))
    if args.cmd == "verify-bundle":
        return _cmd_verify_bundle(args)
    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
