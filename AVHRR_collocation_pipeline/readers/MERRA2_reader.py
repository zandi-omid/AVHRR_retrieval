"""
MERRA2 Reader and Collocation Tools
-----------------------------------

Fast collocation of AVHRR pixels with MERRA2 hourly fields
from daily MERRA2 files (.nc or .nc4).

Author: Omid Zandi
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
from netCDF4 import Dataset

import AVHRR_collocation_pipeline.utils as utils


__all__ = ["collocate_MERRA2", "MissingMERRA2File"]

class MissingMERRA2File(RuntimeError):
    """Raised when required MERRA2 daily file is missing for an orbit."""
    pass

def collocate_MERRA2(
    df,
    MERRA2_meta: Dict[str, Any],
    MERRA2_vars: List[str],
    *, 
    orbit_tag: str | None = None,
    hour_col: str = "scan_hour_m2",
    date_col: str = "scan_date_m2",
    debug: bool = False,
):
    """
    Collocate MERRA2 variables onto the AVHRR DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
          - lon, lat
          - date_col (e.g., "YYYY-MM-DD")
          - hour_col (0..23)  (nearest-hour already decided outside)
    MERRA2_meta : dict
        Output of load_MERRA2_reference(), with keys:
          - date_to_file: dict[str,str]
          - lon: np.ndarray
          - lat: np.ndarray
          - lon_sort_index: np.ndarray | None
          - needs_lat_flip: bool
    MERRA2_vars : list[str]
        Variables to read from MERRA2 files (e.g., ["T2M", "TQV"])
    hour_col, date_col : str
        Column names for hour + date
    prefix : str
        Output columns will be prefix + varname
    debug : bool
        Print helpful diagnostics.

    Returns
    -------
    pandas.DataFrame
        Copy of df with new columns added.
    """
    # ---- sanity checks (fail fast with clear message) ----
    if hour_col not in df.columns:
        raise KeyError(f"'{hour_col}' not found in df columns. You must create it before collocation.")
    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' not found in df columns. You must create it before collocation.")

    date_to_file = MERRA2_meta["date_to_file"]
    merra_lon = MERRA2_meta["lon"]
    merra_lat = MERRA2_meta["lat"]
    lon_sort: Optional[np.ndarray] = MERRA2_meta.get("lon_sort_index", None)
    needs_lat_flip = bool(MERRA2_meta.get("needs_lat_flip", False))

    # Pull arrays once
    LON = df["lon"].to_numpy()
    LAT = df["lat"].to_numpy()

    # Spatial mapping once
    IX, IY = utils.index_finder(LON, LAT, merra_lon, merra_lat)
    ON_GRID = (IX >= 0) & (IY >= 0)

    # Hours and dates
    HOURS = df[hour_col].to_numpy().astype(np.int16)
    DATES = df[date_col].astype(str).to_numpy()

    # Prepare outputs
    out = {f"{v}": np.full(len(df), np.nan, dtype="float32") for v in MERRA2_vars}

    # Group by unique dates (open each file once)
    uniq_dates = np.unique(DATES)

    if debug:
        keys_sample = list(date_to_file.keys())[:5]
        print("\nMERRA2 debug:")
        print("  df scan_date sample:", DATES[0], type(DATES[0]))
        print("  df scan_date unique (first 5):", uniq_dates[:5])
        print("  meta keys (first 5):", keys_sample)
        print("  AVHRR lon min/max:", float(np.nanmin(LON)), float(np.nanmax(LON)))
        print("  AVHRR lat min/max:", float(np.nanmin(LAT)), float(np.nanmax(LAT)))
        print("  MERRA2 lon min/max:", float(np.nanmin(merra_lon)), float(np.nanmax(merra_lon)))
        print("  MERRA2 lat min/max:", float(np.nanmin(merra_lat)), float(np.nanmax(merra_lat)))
        print("  lon_sort_index is None?:", lon_sort is None)
        print("  needs_lat_flip:", needs_lat_flip)

    for date in uniq_dates:
        file_path = date_to_file.get(date)

        if not file_path:
            msg = f"[MERRA2] Missing file for date={date}"
            if orbit_tag is not None:
                msg += f" | orbit={orbit_tag}"
            raise MissingMERRA2File(msg)

        mask_d = (DATES == date) & ON_GRID
        if not np.any(mask_d):
            continue

        # Subset indices for this date (smaller arrays => faster indexing)
        ix_d = IX[mask_d]
        iy_d = IY[mask_d]
        hr_d = HOURS[mask_d]

        # Unique hours needed for this date
        uniq_hr = np.unique(hr_d)

        with Dataset(file_path) as nc:
            for v in MERRA2_vars:
                if v not in nc.variables:
                    if debug:
                        print(f"  WARN: var '{v}' not in file {file_path}")
                    continue

                var = nc[v]  # (time, lat, lon)

                # Load only required hours: (nuniq, lat, lon)
                block = np.asarray(var[uniq_hr, :, :], dtype="float32")

                # If the file lat orientation requires flip (rare; your SUB files don't)
                if needs_lat_flip:
                    block = block[:, ::-1, :]

                # If the file lon grid is 0..360 and we shifted coords to -180..180,
                # we MUST reorder the data to match that shift.
                if lon_sort is not None:
                    block = block[:, :, lon_sort]

                # Map each row’s hour -> block index (vectorized)
                pos = np.searchsorted(uniq_hr, hr_d)

                # Final gather
                out[f"{v}"][mask_d] = block[pos, iy_d, ix_d]

    df2 = df.copy()
    for k, arr in out.items():
        df2[k] = arr
    return df2