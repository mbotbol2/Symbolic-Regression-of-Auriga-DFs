"""Paths, constants and the CV test curves shared by the figure scripts."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..io.auriga_root import resolve_auriga_data_root
from ..io.layout import resolve_halo_dir
from ..io.results_paths import cv_dir, halo_results_root, ids_dir
from ..io.subset import load_subset_phase_space_at_snap

G = 4.30091e-6                  # kpc (km/s)^2 / M_sun
EPS_LO, EPS_HI = 0.03, 0.999    # eps_hat range shown on every DF plot
REF_HALO = 2
SNAP = 127


def out_dir() -> Path:
    """Where figure data is written: $FIG_OUT, else paper_figures/output."""
    out = Path(os.getenv("FIG_OUT", Path(__file__).resolve().parent / "output"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def density_fit(hid: int) -> dict:
    p = halo_results_root() / f"Halo_{hid:02d}" / "density" / "density_fit_params_with_hubble_flow.json"
    return json.loads(p.read_text())


def load_cv():
    """The held-out test curves of every halo on the folds.json grid.

    Returns (grid, raw, offsets): raw[h] is log10 f of halo h in its own units and
    raw[h] + offsets[h] is the same curve in the reference halo's frame.
    """
    cv = cv_dir()
    man = json.loads((cv / "folds.json").read_text())
    grid = np.linspace(man["grid_lo"], man["grid_hi"], man["n_grid"])
    offsets = {int(k): float(v) for k, v in man["to_ref_offset_dex"].items()}
    raw = {}
    for fold in man["folds"]:
        for h in fold["test_halos"]:
            t = pd.read_csv(cv / fold["dir"] / "test" / f"test_halo{h:02d}.csv")
            assert np.allclose(t["eps_hat"], grid, atol=1e-9)
            raw[h] = t["log10f"].to_numpy()
    return grid, dict(sorted(raw.items())), offsets


def bound_particles(hid: int) -> dict:
    """Positions (kpc), velocities (km/s) and masses of halo hid's bound particles at z = 0,
    relative to the central subhalo. Needs $AURIGA_DATA."""
    ids = np.sort(np.load(ids_dir(hid) / "bound_ids_with_hubble_flow_subhalo.npy").astype(np.uint64))
    return load_subset_phase_space_at_snap(
        halo_dir=resolve_halo_dir(resolve_auriga_data_root(), hid), subset_ids_sorted=ids,
        snap=SNAP, pt=1, center_mode="subhalo", apply_vel_sqrt_a=True)
