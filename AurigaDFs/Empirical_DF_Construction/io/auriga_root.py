from __future__ import annotations

import os
from pathlib import Path


def resolve_auriga_data_root() -> Path:
    """
    Resolve the mounted Auriga data root.

    Requires the environment variable AURIGA_DATA to point to the
    directory containing Halo {j}/Z0/, etc.
    """

    env = os.getenv("AURIGA_DATA")
    if not env:
        raise FileNotFoundError(
            "AURIGA_DATA is not set.\n"
            "Set it to the root of your Auriga dataset, e.g.\n"
            "export AURIGA_DATA=/path/to/Auriga"
        )

    root = Path(env).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"AURIGA_DATA points to a non-existent directory:\n{root}"
        )

    return root

