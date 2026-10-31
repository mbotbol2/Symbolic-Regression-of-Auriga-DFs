from __future__ import annotations

from pathlib import Path


def _first_existing_dir(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists() and p.is_dir():
            return p
    return None


def resolve_data_dir_for_snap(halo_dir: Path, snap: int) -> Path:
    """
    Return the directory that contains both Groups/ and Snapshots/ for this halo+snap.

    Supports both layouts:
    - Halo X/{Groups,Snapshots}/... (flat)
    - Halo X/Z*/{Groups,Snapshots}/... (legacy Auriga-style)
    - Halo_{i}/Z{j}/{Groups,Snapshots}/... (your usual layout; we discover the right Z{j} by the snapshot number)
    """
    from .files import groupcat_files, snapshot_files

    halo_dir = Path(halo_dir).expanduser().resolve()
    snap = int(snap)

    has_z = any(p.is_dir() and p.name.startswith("Z") for p in halo_dir.iterdir())

    best: Path | None = None
    best_score = -1
    for zdir in sorted([p for p in halo_dir.iterdir() if p.is_dir() and p.name.startswith("Z")]):
        gdir = _first_existing_dir(zdir / "Groups", zdir / "groups", zdir / "Groups ")
        sdir = _first_existing_dir(zdir / "Snapshots", zdir / "snapshots")
        if gdir is None or sdir is None:
            continue
        n_s = len(snapshot_files(zdir, snap))
        n_g = len(groupcat_files(zdir, snap))
        score = (10 if n_s > 0 else 0) + (1 if n_g > 0 else 0)
        if score > best_score:
            best_score = score
            best = zdir

    if best is None:
        # Fall back to flat layout only when Z* is absent.
        if not has_z:
            gdir = _first_existing_dir(halo_dir / "Groups", halo_dir / "groups", halo_dir / "Groups ")
            sdir = _first_existing_dir(halo_dir / "Snapshots", halo_dir / "snapshots")
            if gdir is not None and sdir is not None:
                return halo_dir
        raise FileNotFoundError(f"Could not find Groups/Snapshots for snap={snap} under {halo_dir}")
    return best


def resolve_halo_dir(base_dir: Path, hid: int) -> Path:
    """
    Resolve a halo directory under base_dir, supporting:
      - Halo {i}       (e.g. $AURIGA_DATA/Halo 2)
      - Halo_{i}       (e.g. Halo_2)
      - Halo_{i:02d}   (e.g. $AURIGA_DATA/Halo_02)
    """
    base_dir = Path(base_dir).expanduser().resolve()
    hid = int(hid)
    candidates = [
        base_dir / f"Halo {hid}",
        base_dir / f"Halo_{hid:02d}",
        base_dir / f"Halo_{hid}",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return candidates[0]

