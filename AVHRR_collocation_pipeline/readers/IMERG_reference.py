from __future__ import annotations
import os
import numpy as np
import datetime
import h5py
import calendar


def list_IMERG_files(IMERG_dir: str) -> np.ndarray:
    files = [
        os.path.join(IMERG_dir, f)
        for f in os.listdir(IMERG_dir)
        if f.endswith(".HDF5") and f.startswith("3B-HHR")
    ]
    if len(files) == 0:
        raise FileNotFoundError(f"No IMERG HDF5 found in: {IMERG_dir}")
    return np.array(sorted(files))


def IMERG_datetime_from_filename(fname: str) -> datetime.datetime:
    base = os.path.basename(fname)
    parts = base.split(".")
    date_part = next(p for p in parts if "-S" in p)

    ymd, start = date_part.split("-S")
    y, m, d = int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8])
    hh = int(start[0:2])
    mm = int(start[2:4])
    ss = int(start[4:6])

    # IMPORTANT: treat IMERG filename time as UTC
    return datetime.datetime(y, m, d, hh, mm, ss, tzinfo=datetime.timezone.utc)


def load_IMERG_grid(sample_file: str):
    with h5py.File(sample_file, "r") as h5:
        lon_raw = h5["/Grid"]["lon"][:] - 0.05
        lat_raw = h5["/Grid"]["lat"][:][::-1] + 0.05

    lon = np.round(lon_raw.astype(float), 1)
    lat = np.round(lat_raw.astype(float), 1)
    return lon, lat


def load_IMERG_reference(IMERG_dir: str):
    files = list_IMERG_files(IMERG_dir)

    dt_list = [IMERG_datetime_from_filename(f) for f in files]
    dt_array = np.array(dt_list)

    # IMPORTANT: robust UTC unix seconds
    dt_stamps = np.array([calendar.timegm(dt.utctimetuple()) for dt in dt_list], dtype="int64")

    lon, lat = load_IMERG_grid(files[0])
    return files, dt_array, dt_stamps, lon, lat