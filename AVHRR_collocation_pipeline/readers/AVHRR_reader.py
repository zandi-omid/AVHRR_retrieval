"""
AVHRR Orbit Reader
===================

This module provides a robust, modular reader for AVHRR Level-2 orbits.
The main function `read_AVHRR_orbit_to_df` loads a single orbit,
applies hemisphere masking, fixes scan-line time wrap-around, grids
the data, and returns a clean pandas DataFrame.

This file is intentionally self-contained for clarity and testing:
all private helper functions begin with `_` and are local to this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import datetime
from typing import Tuple, Dict, List
from netCDF4 import Dataset
import calendar


import AVHRR_collocation_pipeline.utils as utils

__all__ = ["read_AVHRR_orbit_to_df"]


# ---------------------------------------------------------------------
# 1. Read raw latitude, longitude, scan_line_time
# ---------------------------------------------------------------------
def _read_raw_core(nc: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read raw latitude, longitude, and scan_line_time from a NetCDF dataset.

    Returns
    -------
    lat_raw : 2D array (float)
    lon_raw : 2D array (float)
    slt_raw : 1D array (float)
    """
    lat_var = nc["latitude"][:]
    lon_var = nc["longitude"][:]
    slt_var = nc["scan_line_time"][:]

    lat_raw = np.where(lat_var.mask, np.nan, lat_var.data).astype(float)
    lon_raw = np.where(lon_var.mask, np.nan, lon_var.data).astype(float)
    slt_raw = np.where(slt_var.mask, np.nan, slt_var.data).astype(float)

    return lat_raw, lon_raw, slt_raw


# ---------------------------------------------------------------------
# 2. Hemisphere masking (keep only lat ≥ Nthresh or lat ≤ Sthresh)
# ---------------------------------------------------------------------
def _apply_hemisphere_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    north_thresh: float,
    south_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mask out mid-latitudes between thresholds. AVHRR often requires
    restricting to poleward regions (e.g., ±55°).

    Returns
    -------
    lat_masked : 2D array
    lon_masked : 2D array
    mask       : boolean 2D array where True indicates invalid pixels
    """
    hemi_mask = (lat < north_thresh) & (lat > south_thresh)
    lat_m = np.where(hemi_mask, np.nan, lat)
    lon_m = np.where(np.isnan(lat_m), np.nan, lon)
    mask = np.isnan(lat_m)
    return lat_m, lon_m, mask


# ---------------------------------------------------------------------
# 3. Fix single midnight wrap-around
# ---------------------------------------------------------------------
def _fix_scanline_wraparound(
    slt: np.ndarray,
    nd: int,
    avh_file: str
) -> np.ndarray:
    """
    Fix midnight wrap-around for 1-day orbits.
    Raises ValueError if multiple negative jumps are found.

    Returns
    -------
    slt_fixed : 1D array (float)
    """
    slt = slt.astype(float).copy()

    if nd != 1:
        return slt

    diffs = np.diff(slt)
    neg_locs = np.where(diffs < 0)[0]

    if len(neg_locs) == 0:
        return slt

    if len(neg_locs) > 1:
        msg = (
            f"ERROR: multiple wrap-arounds detected in scan_line_time:\n"
            f"  {avh_file}\n"
            f"Orbit appears broken."
        )
        print(msg)
        raise ValueError(msg)

    idx = neg_locs[0]
    slt[idx + 1:] += 24.0
    return slt


# ---------------------------------------------------------------------
# 4. Convert scan_line_time hours → UNIX timestamps
# ---------------------------------------------------------------------
def _convert_scanline_to_unix(slt: np.ndarray, base_dt: datetime.datetime) -> np.ndarray:
    slt = slt.astype("float64")
    ts_out = np.full_like(slt, np.nan, dtype="float64")

    valid = np.isfinite(slt)
    if not valid.any():
        return ts_out

    # Treat base_dt as UTC midnight (same intent as old code)
    if base_dt.tzinfo is None:
        base_dt = base_dt.replace(tzinfo=datetime.timezone.utc)

    base_epoch = calendar.timegm(base_dt.timetuple())  # int seconds

    # OLD behavior: int(timestamp()) => truncation to whole seconds
    # Use floor to reproduce "01:29:59.999 -> 01:29:59"
    ts = base_epoch + slt[valid] * 3600.0

    # tiny epsilon helps when ts is like 5399.999999999 due to float rep
    ts_int = np.floor(ts + 1e-9).astype("int64")

    ts_out[valid] = ts_int.astype("float64")
    return ts_out
# ---------------------------------------------------------------------
# 5. Read requested AVHRR variables
# ---------------------------------------------------------------------


def _read_AVHRR_variables(
    nc: Dataset,
    base_mask: np.ndarray,
    var_list: List[str],
    ignore_missing: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Read and mask AVHRR variables safely.

    Parameters
    ----------
    nc : Dataset
        Open NetCDF4 AVHRR orbit file.
    base_mask : np.ndarray
        Mask from lat/lon representing invalid pixels.
    var_list : list[str]
        Variables to read (e.g., ["temp_12_0um_nom", "reflectance_ch1"])
    ignore_missing : bool
        If True: skip variables not found.
        If False: raise an error when a variable is missing.

    Returns
    -------
    dict
        {variable_name: masked 2D float array}
    """

    out = {}

    for var in var_list:

        # -----------------------------------------
        # Validate variable existence
        # -----------------------------------------
        if var not in nc.variables:
            msg = f"Variable '{var}' not found in NetCDF file: {nc.filepath()}"
            if ignore_missing:
                print(f"WARNING: {msg} — skipping.")
                continue
            else:
                raise KeyError(msg)

        # -----------------------------------------
        # Read variable safely
        # -----------------------------------------
        var_nc = nc[var][:]
        arr = np.where(var_nc.mask, np.nan, var_nc.data).astype(float)
        arr = np.where(base_mask, np.nan, arr)

        out[var] = arr

    return out

# ---------------------------------------------------------------------
# 6. Build unified valid-pixel mask
# ---------------------------------------------------------------------
def _build_valid_mask(
    lat_f,
    lon_f,
    time_f,
    var_f_dict,
):
    """
    Require all fields (lat, lon, time, variables) to be non-NaN.

    Returns
    -------
    mask : boolean 1D array
    """
    mask = (~np.isnan(lat_f)) & (~np.isnan(lon_f)) & (~np.isnan(time_f))
    for arr in var_f_dict.values():
        mask &= ~np.isnan(arr)
    return mask


# ---------------------------------------------------------------------
# 7. Grid & aggregate by grid cell
# ---------------------------------------------------------------------
def _grid_and_aggregate(
    lat_f,
    lon_f,
    time_f,
    var_f_dict,
    x_vec,
    y_vec,
    x,
    y
):
    """
    Map flattened AVHRR pixels to grid (using index_finder),
    aggregate by grid cell, and attach lon/lat from the target grid.

    Returns
    -------
    dfm : pandas DataFrame
    """
    idx, idy = utils.index_finder(lon_f, lat_f, x_vec, y_vec)
    on_grid = (idx >= 0) & (idy >= 0)

    if not on_grid.any():
        raise ValueError("All valid pixels fall outside the target grid.")

    df_dict = {
        "idx": idx[on_grid],
        "idy": idy[on_grid],
        "scan_line_times": time_f[on_grid],
    }
    for k, v in var_f_dict.items():
        df_dict[k] = v[on_grid]

    df = pd.DataFrame(df_dict)
    dfm = df.groupby(["idx", "idy"], as_index=False).mean()

    dfm["lon"] = x[dfm["idy"], dfm["idx"]]
    dfm["lat"] = y[dfm["idy"], dfm["idx"]]

    return dfm


# =====================================================================
#                      PUBLIC ENTRY POINT
# =====================================================================
def read_AVHRR_orbit_to_df(
    avh_file: str,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    avh_vars: list[str],
    lat_thresh_N_hemisphere: float = 55.0,
    lat_thresh_S_hemisphere: float = -55.0,
) -> pd.DataFrame:
    """
    High-level reader for a single AVHRR Level-2 orbit.

    Parameters
    ----------
    avh_file : str
        Path to NetCDF file.
    x_vec, y_vec : 1D arrays
        Grid vectors used for pixel→grid mapping.
    x, y : 2D arrays
        Full grid (same shape), used to assign output cell coordinates.
    avh_vars : list[str]
        Variable names from the AVHRR file to load & aggregate.
    lat_thresh_N_hemisphere : float
        Northern cutoff.
    lat_thresh_S_hemisphere : float
        Southern cutoff.

    Returns
    -------
    dfm : pandas.DataFrame
        Columns: ['lon', 'lat', 'scan_line_times'] + avh_vars
    """
    # Parse times from filename
    avh_st, avh_et, _ = utils.AVHRR_datetime_NC_files(avh_file)
    nd = (avh_et.date() - avh_st.date()).days

    with Dataset(avh_file) as nc:
        # (1) core arrays
        lat_raw, lon_raw, slt_raw = _read_raw_core(nc)

        # (2) hemisphere masking
        lat_m, lon_m, hemi_mask = _apply_hemisphere_mask(
            lat_raw, lon_raw,
            lat_thresh_N_hemisphere,
            lat_thresh_S_hemisphere,
        )

        # (3) fix wrap-around
        slt_fixed = _fix_scanline_wraparound(slt_raw, nd, avh_file)

        # (4) convert to UNIX timestamps
        base_dt = avh_st.replace(hour=0, minute=0, second=0)
        ts_1d = _convert_scanline_to_unix(slt_fixed, base_dt)
        ts_2d = np.tile(ts_1d.reshape(-1, 1), lat_m.shape[1])

        # (5) read requested variables
        var_arrays = _read_AVHRR_variables(nc, hemi_mask, avh_vars)

    # (6) flatten
    lat_f = lat_m.ravel()
    lon_f = lon_m.ravel()
    time_f = ts_2d.ravel()
    var_f = {k: v.ravel() for k, v in var_arrays.items()}

    # (7) build unified mask
    valid = _build_valid_mask(lat_f, lon_f, time_f, var_f)
    if not valid.any():
        raise ValueError(f"No valid pixels found in orbit: {avh_file}")

    # apply mask
    lat_f = lat_f[valid]
    lon_f = lon_f[valid]
    time_f = time_f[valid]
    var_f = {k: v[valid] for k, v in var_f.items()}

    # (8) grid & aggregate
    dfm = _grid_and_aggregate(
        lat_f, lon_f, time_f, var_f,
        x_vec, y_vec, x, y,
    )

    # reorder
    dfm = dfm.loc[:, ["lon", "lat", "scan_line_times"] + avh_vars]
    return dfm