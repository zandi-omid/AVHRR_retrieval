from __future__ import annotations
import numpy as np
import h5py
import AVHRR_collocation_pipeline.utils as utils


def collocate_IMERG_precip(df, IMERG_meta):
    (
        IMERG_FILES,
        IMERG_DT,
        IMERG_DT_STAMPS,
        IMERG_LON,
        IMERG_LAT,
    ) = IMERG_meta

    # ---- choose scan times ----
    scan_times = df["scan_line_times"].to_numpy(dtype="int64")
    IMERG_DT_STAMPS = np.asarray(IMERG_DT_STAMPS, dtype="int64")
    sel_idx = np.searchsorted(IMERG_DT_STAMPS, scan_times, side="right") - 1

    # ---- mark out-of-range as invalid (do NOT raise) ----
    valid_t = (sel_idx >= 0) & (sel_idx < len(IMERG_FILES))
    n_bad = np.count_nonzero(~valid_t)
    if n_bad:
        print(f"[IMERG] {n_bad}/{len(df)} pixels outside IMERG time range -> IMERG_preci=NaN")

    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    idx, idy = utils.index_finder(lon, lat, IMERG_LON, IMERG_LAT)
    good_xy = (idx >= 0) & (idy >= 0)

    good = valid_t & good_xy

    IMERG_out = np.full(len(df), np.nan, dtype="float32")

    # only loop over indices that actually occur in valid points
    uniq_indices = np.unique(sel_idx[good])

    for t_idx in uniq_indices:
        mask_t = (sel_idx == t_idx) & good
        if not np.any(mask_t):
            continue

        IMERG_file = IMERG_FILES[int(t_idx)]
        with h5py.File(IMERG_file, "r") as h5:
            arr = h5["Grid/precipitation"][0].transpose()
            arr = np.flip(arr, axis=0)
            arr = np.where(arr == -9999.9, np.nan, arr)

        IMERG_out[mask_t] = arr[idy[mask_t], idx[mask_t]]

    df2 = df.copy()
    df2["IMERG_preci"] = IMERG_out
    return df2