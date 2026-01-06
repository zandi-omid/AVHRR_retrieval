"""
NetCDF encoding utilities
-------------------------

Central place for xarray -> NetCDF encoding conventions.

We mostly store float fields as uint16 with scale_factor/add_offset
for good compression + compact storage.

Author: Omid Zandi
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import xarray as xr


def enc_uint16_sf(
    *,
    scale_factor: float = 0.005,
    add_offset: float = 0.0,
    fill_value: int = 65535,
    complevel: int = 9,
) -> Dict:
    """
    Standard packed uint16 encoding for floats.

    Notes:
    - xarray handles scale_factor/add_offset packing.
    - _FillValue must be uint16.
    """
    return dict(
        zlib=True,
        complevel=int(complevel),
        shuffle=True,
        dtype="uint16",
        scale_factor=float(scale_factor),
        add_offset=float(add_offset),
        _FillValue=np.uint16(fill_value),
    )


def build_uint16_encoding(
    ds: xr.Dataset,
    *,
    default_scale: float = 0.005,
    var_scales: Optional[Dict[str, float]] = None,
    var_offsets: Optional[Dict[str, float]] = None,
    complevel: int = 9,
) -> Dict[str, Dict]:
    """
    Build an encoding dict for all data variables in a dataset.

    Example:
      build_uint16_encoding(ds, var_scales={"T2M": 0.1, "TQV": 0.01})

    Parameters
    ----------
    default_scale : float
        scale_factor for variables not listed in var_scales
    var_scales : dict[str, float] | None
        per-variable scale_factor overrides
    var_offsets : dict[str, float] | None
        per-variable add_offset overrides (usually 0)
    """
    var_scales = var_scales or {}
    var_offsets = var_offsets or {}

    enc: Dict[str, Dict] = {}
    for v in ds.data_vars:
        sf = var_scales.get(v, default_scale)
        off = var_offsets.get(v, 0.0)
        enc[v] = enc_uint16_sf(scale_factor=sf, add_offset=off, complevel=complevel)

    return enc