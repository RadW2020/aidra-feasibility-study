"""Keep the model cards in the models volume in step with git.

On the Coolify deployment ``/app/models`` is a named volume that shadows the
image's ``models/`` directory: Docker seeded it from the image on the first
deploy and never again, so cards added or corrected in git stopped reaching
the server (and ``ModelManager._require_model_card`` refuses to load a weight
without its card, I-MOD-4 / I-AIA-1). The Dockerfile now also copies the
cards to ``/app/models_dist/cards`` (outside the volume) and the lifespan
calls :func:`sync_model_cards` before the registry scan.

Rule: git is the source of truth for cards. A card that exists in the volume
with different content is **backed up** next to itself as
``<name>.superseded-<UTC timestamp>`` before being overwritten — mark, never
delete (CLAUDE.md §8). Files that only exist in the volume are left alone.
"""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CARD_SUFFIX = ".MODEL_CARD.md"


def sync_model_cards(dist_dir: Path | str, dest_dir: Path | str) -> dict[str, list[str]]:
    """Copy cards from ``dist_dir`` (image) into ``dest_dir`` (volume).

    Returns ``{"added": [...], "updated": [...], "unchanged": [...], "backups": [...]}``
    with card file names. Missing ``dist_dir`` is a no-op (local dev runs
    straight from the repo, where both paths are the same directory).
    """
    dist, dest = Path(dist_dir), Path(dest_dir)
    result: dict[str, list[str]] = {"added": [], "updated": [], "unchanged": [], "backups": []}
    if not dist.is_dir():
        logger.debug("cards_sync: no dist dir at %s — skipping", dist)
        return result
    if dist.resolve() == dest.resolve():
        return result
    dest.mkdir(parents=True, exist_ok=True)
    for card in sorted(dist.glob(f"*{CARD_SUFFIX}")):
        target = dest / card.name
        if not target.exists():
            shutil.copy2(card, target)
            result["added"].append(card.name)
            continue
        if target.read_bytes() == card.read_bytes():
            result["unchanged"].append(card.name)
            continue
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = dest / f"{card.name}.superseded-{stamp}"
        shutil.copy2(target, backup)
        shutil.copy2(card, target)
        result["backups"].append(backup.name)
        result["updated"].append(card.name)
    if result["added"] or result["updated"]:
        logger.info(
            "cards_sync: %d added, %d updated (%d backed up), %d unchanged (%s -> %s)",
            len(result["added"]), len(result["updated"]), len(result["backups"]),
            len(result["unchanged"]), dist, dest,
        )
    return result
