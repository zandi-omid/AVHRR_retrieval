# AVHRR_collocation_pipeline/retrievers/limb_correction/lut_loader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import pickle
from typing import Dict, Any

import pandas as pd

# Simple in-process cache (important for repeated calls on same rank)
_ASSET_CACHE: dict[str, "LimbCorrAssets"] = {}


@dataclass(frozen=True)
class LimbCorrAssets:
    """
    Container for all limb-correction assets:
      - LUT tables (temp_11/temp_12 × NH/SH × season)
      - global geometry curves
      - surface-specific geometry curves
      - constants like NADIR_CENTER and surface_type_mapping
    """
    path_to_lut: str
    all_lut: Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
    global_curve: Dict[str, Any]
    curve_lib: Dict[str, Any]
    surface_type_mapping: Dict[int, str]
    NADIR_CENTER: int = 204


def load_limbcorr_assets(path_to_lut: str, use_cache: bool = True) -> LimbCorrAssets:
    """
    Load LUTs + geometry curve pickles from disk ONCE.
    Returns LimbCorrAssets used by correction.py.

    Parameters
    ----------
    path_to_lut : str
        Directory containing:
          - temp_11_<NH/SH>_<Season>_*.pkl (your naming pattern)
          - temp_12_<NH/SH>_<Season>_*.pkl
          - global_geometry.pkl
          - surface_geometry.pkl
    use_cache : bool
        Cache assets per process to avoid repeated load.

    Returns
    -------
    LimbCorrAssets
    """
    p = str(Path(path_to_lut).resolve())
    if use_cache and p in _ASSET_CACHE:
        return _ASSET_CACHE[p]

    lut_dir = Path(p)
    if not lut_dir.exists():
        raise FileNotFoundError(f"LUT directory not found: {p}")

    # Read all LUTs
    all_lut_files = sorted(
        str(lut_dir / s)
        for s in os.listdir(lut_dir)
        if s.startswith("temp") and s.endswith(".pkl")
    )

    all_lut: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}

    for file_path in all_lut_files:
        file_name = os.path.basename(file_path)
        parts = file_name.split("_")
        if len(parts) < 4:
            # unexpected naming; skip safely
            continue

        # Example expected:
        # temp_11_NH_Winter_*.pkl
        var = parts[0] + "_" + parts[1]         # temp_11 / temp_12
        hemisphere = parts[2]                   # NH / SH
        season = parts[3]                       # Winter / Summer / ...

        all_lut.setdefault(var, {}).setdefault(hemisphere, {})

        df = pd.read_pickle(file_path)
        all_lut[var][hemisphere][season] = df

    # Load geometry curves
    global_geom = lut_dir / "global_geometry.pkl"
    surface_geom = lut_dir / "surface_geometry.pkl"

    if not global_geom.exists():
        raise FileNotFoundError(f"Missing {global_geom}")
    if not surface_geom.exists():
        raise FileNotFoundError(f"Missing {surface_geom}")

    with open(global_geom, "rb") as f:
        global_curve = pickle.load(f)

    with open(surface_geom, "rb") as f:
        curve_lib = pickle.load(f)

    surface_type_mapping = {
        0: "water",
        1: "snow-free land",
        2: "snow-covered land",
        3: "ice",
    }

    assets = LimbCorrAssets(
        path_to_lut=p,
        all_lut=all_lut,
        global_curve=global_curve,
        curve_lib=curve_lib,
        surface_type_mapping=surface_type_mapping,
    )

    if use_cache:
        _ASSET_CACHE[p] = assets

    return assets