from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import h5py


@dataclass(frozen=True)
class HeaderInfo:
    a: float
    h: float
    z: float
    omega_m: float
    omega_l: float
    mass_table: np.ndarray | None


def read_header(snapshot_file: Path) -> HeaderInfo:
    snapshot_file = Path(snapshot_file)
    with h5py.File(snapshot_file, "r") as f:
        hdr = f["Header"].attrs
        a = float(hdr.get("Time", 1.0))
        h = float(hdr.get("HubbleParam", np.nan))
        z = float(hdr.get("Redshift", (1.0 / a - 1.0) if a else np.nan))
        om = float(hdr.get("Omega0", np.nan))
        ol = float(hdr.get("OmegaLambda", np.nan))
        mt = hdr.get("MassTable", None)
        mass_table = np.asarray(mt, dtype=float) if mt is not None else None
    return HeaderInfo(a=a, h=h, z=z, omega_m=om, omega_l=ol, mass_table=mass_table)

