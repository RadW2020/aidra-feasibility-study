"""
Build the D3 Evidence Bundle.

Convenience wrapper around ``python -m src.traceability bundle`` that
pre-fills the ``bundle`` subcommand so callers don't have to repeat it.
Prints the path to the generated bundle on stdout, which is consumed by
the ``/build-evidence-bundle`` slash command and by CI scripts that
archive artifacts.

Usage::

    python -m scripts.build_d3_bundle --out /data/evidence
    python -m scripts.build_d3_bundle --out /data/evidence --zone gibraltar
    python -m scripts.build_d3_bundle --out /data/evidence \
        --model vesseltracker-sar-yolov8 --no-archive

Output:
    /data/evidence/d3-<timestamp>.tar.gz   (default)
    /data/evidence/d3-<timestamp>/         (--no-archive)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.traceability.__main__ import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    # Pre-pend the 'bundle' subcommand so callers don't need to repeat it.
    if not argv or argv[0] not in {"bundle", "verify", "verify-bundle", "-h", "--help"}:
        argv = ["bundle", *argv]
    raise SystemExit(main(argv))
