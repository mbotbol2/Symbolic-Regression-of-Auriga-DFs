from __future__ import annotations

from pathlib import Path

from .layout import _first_existing_dir


def _find(top: Path | None, subdir: str, stem: str, snap: int) -> list[Path]:
    """Files <stem>_<snap>.*.hdf5 directly in top or in top/<subdir>_<snap>, with the snapshot
    number zero-padded to three digits first, then unpadded."""
    if top is None:
        return []
    for name in (f"{snap:03d}", f"{snap}"):
        for d in (top, top / f"{subdir}_{snap:03d}"):
            files = sorted(d.glob(f"{stem}_{name}.*.hdf5"))
            if files:
                return files
    return []


def snapshot_files(data_dir: Path, snap: int) -> list[Path]:
    data_dir = Path(data_dir)
    sdir = _first_existing_dir(data_dir / "Snapshots", data_dir / "snapshots")
    return _find(sdir, "snapdir", "snapshot", int(snap))


def groupcat_files(data_dir: Path, snap: int) -> list[Path]:
    data_dir = Path(data_dir)
    gdir = _first_existing_dir(data_dir / "Groups", data_dir / "groups", data_dir / "Groups ")
    return _find(gdir, "groups", "fof_subhalo_tab", int(snap))
