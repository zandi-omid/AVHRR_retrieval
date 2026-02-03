from __future__ import annotations

import numpy as np
import netCDF4 as nc
import AVHRR_collocation_pipeline.utils as utils


def collocate_ERA5_precip(
    df,
    ERA5_meta_by_year,
    varname: str = "tp",
    out_col: str | None = None,
    scale: float = 1000.0,          # tp: meters -> mm
    *,
    time_key: str = "scan_hour_unix_era5",  # default mimics OLD
):
    """
    Collocate ERA5 hourly data (one file per year) to df.

    Timing:
      - Uses df['scan_hour_unix'] (nearest-hour unix, from add_time_columns)
      - Then applies time_offset_seconds (DEFAULT +3600 to mimic OLD pipeline)

    OLD pipeline behavior replicated:
      - nearest-hour rounding + 1 hour shift
      - mask -> NaN
      - negative tp -> 0
    """
    if out_col is None:
        out_col = f"ERA5_{varname}"

    # -----------------------
    # Time (nearest-hour unix)
    # -----------------------
    if time_key in df.columns:
        t_hour = df[time_key].to_numpy().astype("int64")
    elif "scan_hour_unix" in df.columns:
        t_hour = df["scan_hour_unix"].to_numpy().astype("int64")
    else:
        t = df["scan_line_times"].to_numpy().astype("int64")
        t_hour = ((t + 1800) // 3600) * 3600

    # Year per row (after offset)
    years = (
        t_hour.astype("datetime64[s]")
        .astype("datetime64[Y]")
        .astype(int) + 1970
    ).astype(np.int32)

    # -----------------------
    # Spatial indices per row
    # -----------------------
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()

    out = np.full(len(df), np.nan, dtype="float32")

    for yr in np.unique(years):
        meta = ERA5_meta_by_year.get(int(yr))
        if meta is None:
            continue

        m_year = (years == yr)
        if not np.any(m_year):
            continue

        ix, iy = utils.index_finder(lon[m_year], lat[m_year], meta["lon"], meta["lat"])
        good_xy = (ix >= 0) & (iy >= 0)
        if not np.any(good_xy):
            continue

        rows_year = np.where(m_year)[0]
        rows_good = rows_year[good_xy]

        # Exact match on hourly unix
        t_sub = t_hour[m_year][good_xy]
        tidx = np.searchsorted(meta["time_unix"], t_sub, side="left")

        ok_t = (
            (tidx >= 0)
            & (tidx < len(meta["time_unix"]))
            & (meta["time_unix"][tidx] == t_sub)
        )
        if not np.any(ok_t):
            continue

        rows = rows_good[ok_t]
        ix2 = ix[good_xy][ok_t]
        iy2 = iy[good_xy][ok_t]
        tidx2 = tidx[ok_t]

        lon_sort = meta["lon_sort_index"]

        with nc.Dataset(meta["file"]) as ds:
            var = ds[varname]  # (time, lat, lon_org)

            for t_unique in np.unique(tidx2):
                mm = (tidx2 == t_unique)
                rr = rows[mm]
                xs = ix2[mm]
                ys = iy2[mm]

                arr = var[int(t_unique), :, :]      # lon still 0..360 (maybe masked)
                arr = arr[:, lon_sort]              # reorder to -180..180

                # Handle masked arrays -> NaN
                if np.ma.isMaskedArray(arr):
                    arr = arr.filled(np.nan)

                vals = arr[ys, xs]                  # float/ndarray
                vals = (vals * scale).astype("float32")

                vals = np.where(np.isfinite(vals) & (vals < 0), 0.0, vals)

                out[rr] = vals

    df2 = df.copy()
    df2[out_col] = out
    return df2