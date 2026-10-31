from __future__ import annotations

import os
from pathlib import Path

CV_NAME = "cv_with_hubble_flow_nfw_ref02"


def halo_results_root() -> Path:
    """Where pipeline outputs live: $HALO_RESULTS, else AurigaDFs/Halo_Results."""
    env = os.getenv("HALO_RESULTS")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "Halo_Results"


def halo_name(hid: int) -> str:
    return f"Halo_{int(hid):02d}"


def halo_dir(hid: int) -> Path:
    return halo_results_root() / halo_name(hid)


def ids_dir(hid: int) -> Path:
    return halo_dir(hid) / "ids"


def density_dir(hid: int) -> Path:
    return halo_dir(hid) / "density"


def df_dir(hid: int) -> Path:
    return halo_dir(hid) / "df"


def df_raw_dir(hid: int) -> Path:
    return df_dir(hid) / "raw"


def overlays_dir() -> Path:
    return halo_results_root() / "_overlays"


def sr_dir() -> Path:
    """Symbolic-regression datasets: train/test split and bootstrap errors."""
    return halo_results_root() / "_sr"


def cv_dir() -> Path:
    """The cross-validation run the paper uses: $CV_DIR, else <results>/_sr/<CV_NAME>."""
    env = os.getenv("CV_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return sr_dir() / CV_NAME
