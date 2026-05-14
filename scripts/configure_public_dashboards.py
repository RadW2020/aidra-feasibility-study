"""Ensure every public Grafana dashboard exposes the time-range picker.

Grafana stores `timeSelectionEnabled` (and `annotationsEnabled`) in the
`dashboard_public` table — not in the dashboard JSON. Provisioning by file
does not touch those flags, so they have to be set via the admin API.

Usage:
    GRAFANA_URL=https://aidra.uliber.com \
    GRAFANA_USER=admin \
    GRAFANA_PASSWORD=... \
        python scripts/configure_public_dashboards.py

If `.env` exists in repo root, it is loaded automatically.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = REPO_ROOT / "grafana" / "dashboards"
ENV_FILE = REPO_ROOT / ".env"


def _load_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _iter_dashboard_uids() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in sorted(DASHBOARD_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        uid = data.get("uid")
        title = data.get("title", path.stem)
        if uid:
            out.append((uid, title))
    return out


def main() -> int:
    _load_env()
    base = os.environ.get("GRAFANA_URL", "https://aidra.uliber.com").rstrip("/") + "/"
    user = os.environ.get("GRAFANA_USER", "admin")
    pwd = os.environ.get("GRAFANA_PASSWORD")
    if not pwd:
        print("error: GRAFANA_PASSWORD not set (check .env)", file=sys.stderr)
        return 2

    auth = (user, pwd)
    session = requests.Session()
    session.auth = auth
    session.headers.update({"Content-Type": "application/json"})

    failures = 0
    for uid, title in _iter_dashboard_uids():
        pd_url = urljoin(base, f"api/dashboards/uid/{uid}/public-dashboards")
        r = session.get(pd_url, timeout=15)
        if r.status_code == 404:
            print(f"[skip] {title} ({uid}): no public dashboard configured")
            continue
        if not r.ok:
            print(f"[fail] {title} ({uid}): GET {r.status_code} {r.text[:120]}")
            failures += 1
            continue
        cfg = r.json()
        pd_uid = cfg.get("uid")
        already_on = bool(cfg.get("timeSelectionEnabled")) and bool(cfg.get("annotationsEnabled"))
        if already_on:
            print(f"[ok]   {title} ({uid}): timeSelection+annotations already enabled")
            continue

        patch_url = urljoin(base, f"api/dashboards/uid/{uid}/public-dashboards/{pd_uid}")
        body = {"timeSelectionEnabled": True, "annotationsEnabled": True}
        pr = session.patch(patch_url, data=json.dumps(body), timeout=15)
        if pr.ok:
            print(f"[set]  {title} ({uid}): time picker + annotations enabled")
        else:
            print(f"[fail] {title} ({uid}): PATCH {pr.status_code} {pr.text[:200]}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
