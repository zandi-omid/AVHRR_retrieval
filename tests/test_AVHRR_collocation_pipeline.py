import os
import numpy as np

from AVHRR_collocation_pipeline.readers.AVHRR_reader import (
    read_AVHRR_orbit_to_df,
)
from AVHRR_collocation_pipeline.readers.IMERG_reference import (
    load_IMERG_reference,
)
from AVHRR_collocation_pipeline.readers.IMERG_reader import (
    collocate_IMERG_precip,
)
from AVHRR_collocation_pipeline.readers.ERA5_reference import (
    load_ERA5_reference,
)
from AVHRR_collocation_pipeline.readers.ERA5_reader import (
    collocate_ERA5_precip,
)

from AVHRR_collocation_pipeline.readers.AutoSnow_reference import (
    load_AutoSnow_reference,
)
from AVHRR_collocation_pipeline.readers.AutoSnow_reader import (
    collocate_AutoSnow,
)

# ---------------- MERRA2 ----------------
from AVHRR_collocation_pipeline.readers.MERRA2_reference import (
    load_MERRA2_reference,
)
from AVHRR_collocation_pipeline.readers.MERRA2_reader import (
    collocate_MERRA2,
)

# ------------------ add this import near the top ------------------
from AVHRR_collocation_pipeline.reproject import reproject_vars_wgs_to_polar

import AVHRR_collocation_pipeline.utils as utils


# -----------------------------------------
# CONFIG
# -----------------------------------------
AVHRR_FOLDERS = [
    "/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/2010/",
]

OUT_DIR = "/home/omidzandi/test_maps/"
os.makedirs(OUT_DIR, exist_ok=True)

GRID_RES: float = 0.5  # degrees

# Hemisphere split thresholds (also reusable later for reprojection)
LAT_THRESH_NH: float = 55.0
LAT_THRESH_SH: float = -55.0

AVHRR_VARS: list[str] = [
    "cloud_probability",
    "temp_11_0um_nom",
    "temp_12_0um_nom",
]

# IMERG directory (2010 half-hourly V07B)
IMERG_DIR: str = (
    "/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/IMERGV7/Data_V7_halfhourly_2010"
)

# ERA5 directory containing Total_precip_YYYY.nc (0.25°)
ERA5_DIR: str = (
    "/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_0.25deg"
)

# MERRA2 directory with daily hourly files (*.nc4)
MERRA2_DIR: str = (
    "/ra1/pubdat/AVHRR_CloudSat_proj/MERRA2/merra2_archive_19800101_20250831"
)

# Which MERRA2 variables to collocate (must exist inside your daily files)
MERRA2_VARS: list[str] = [
    "TQV",
    "T2M",
]

AUTOSNOW_DIR: str = (
    "/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif"
)

# Which layers to save as GeoTIFF
SAVE_LAYERS: list[str] = [
    "temp_12_0um_nom",
    "IMERG_preci",
    "ERA5_tp",
    "TQV",
    "T2M",
    "AutoSnow",
]

# Polar reprojection controls
DO_REPROJECT_TO_POLAR: bool = True
POLAR_OUT_DIR: str = "/home/omidzandi/test_maps_polar/"
os.makedirs(POLAR_OUT_DIR, exist_ok=True)

LAT_TS_NH: float = 70.0
LAT_TS_SH: float = -71.0

# -----------------------------------------
# LOAD REFERENCES ONCE FOR THE TEST
# -----------------------------------------
IMERG_meta = load_IMERG_reference(IMERG_DIR)
ERA5_meta_by_year = load_ERA5_reference(ERA5_DIR)
MERRA2_meta = load_MERRA2_reference(MERRA2_DIR)
AutoSnow_meta = load_AutoSnow_reference(AUTOSNOW_DIR)


def _print_nan_diagnostics(df, col: str, title: str) -> None:
    n_total = len(df)
    n_ok = int(np.isfinite(df[col]).sum())
    n_nan = int(np.isnan(df[col]).sum())
    print(f"\n{title} diagnostics:")
    print(f"  Total rows:        {n_total}")
    print(f"  Non-NaN values:    {n_ok}")
    print(f"  NaN values:        {n_nan}")


def _save_layer_to_tiff(df, layer: str, avh_file: str, x_vec, y_vec) -> None:
    grid = utils.df2grid(df, layer, x_vec, y_vec)
    out_tif = utils.build_tiff_name(avh_file, layer, OUT_DIR)
    utils.save_grid_to_tiff(grid, out_tif, x_vec, y_vec)
    print(f"Saved {layer} grid → {out_tif}")


def test_AVHRR_collocation() -> None:
    """
    General end-to-end test for AVHRR collocation pipeline:

      1) Read AVHRR orbit into DataFrame
      2) Collocate IMERG precipitation
      3) Collocate ERA5 precipitation
      4) Collocate MERRA2 variables (hourly, one file per day)
      5) Collocate AutoSnow (daily GeoTIFF)
      6) Save selected layers as GeoTIFFs
    """

    # ------------------------------------------------------
    # 1) Pick a random AVHRR file
    # ------------------------------------------------------
    avh_file: str = utils.pick_random_nc_file(AVHRR_FOLDERS)
    print(f"\nSelected AVHRR orbit:\n  {avh_file}\n")

    # ------------------------------------------------------
    # 2) Build target grid for AVHRR (global GRID_RES)
    # ------------------------------------------------------
    x_vec, y_vec, x, y = utils.build_test_grid(GRID_RES)

    # ------------------------------------------------------
    # 3) Read AVHRR orbit into DataFrame
    # ------------------------------------------------------
    df = read_AVHRR_orbit_to_df(
        avh_file,
        x_vec,
        y_vec,
        x,
        y,
        avh_vars=AVHRR_VARS,
        lat_thresh_N_hemisphere=LAT_THRESH_NH,
        lat_thresh_S_hemisphere=LAT_THRESH_SH,
    )

    if df is None or df.empty:
        print("AVHRR reader returned empty or None dataframe. Abort.")
        return

    print("AVHRR DataFrame (before IMERG/ERA5/MERRA2):")
    print(df.head())
    print(f"\nNumber of AVHRR pixels: {len(df)}\n")

    df = utils.add_time_columns(df)

    # ------------------------------------------------------
    # 4) Collocate IMERG precipitation
    # ------------------------------------------------------
    df = collocate_IMERG_precip(df, IMERG_meta)
    print("\nAVHRR + IMERG (sample rows):")
    print(df[["lon", "lat", "scan_line_times", "IMERG_preci"]].head())
    _print_nan_diagnostics(df, "IMERG_preci", "IMERG")

    # ------------------------------------------------------
    # 5) Collocate ERA5 total precipitation
    # ------------------------------------------------------
    df = collocate_ERA5_precip(df, ERA5_meta_by_year, varname="tp")
    era5_col = "ERA5_tp"
    print("\nAVHRR + IMERG + ERA5 (sample rows):")
    print(df[["lon", "lat", "scan_line_times", "IMERG_preci", era5_col]].head())
    _print_nan_diagnostics(df, era5_col, "ERA5")

    # ------------------------------------------------------
    # 6) Collocate MERRA2 (hourly, one file per day)
    # ------------------------------------------------------
    if len(MERRA2_VARS) == 0:
        print("\nMERRA2 skipped (MERRA2_VARS is empty).")
    else:
        df = collocate_MERRA2(df, MERRA2_meta, MERRA2_vars=MERRA2_VARS)

        print("\nAVHRR + IMERG + ERA5 + MERRA2 (sample rows):")
        show_cols = ["lon", "lat", "scan_line_times", "IMERG_preci", era5_col]
        show_cols += [v for v in MERRA2_VARS if v in df.columns]
        print(df[show_cols].head())

        for v in MERRA2_VARS:
            if v in df.columns:
                _print_nan_diagnostics(df, v, f"MERRA2 {v}")

    # ------------------------------------------------------
    # 7) Collocate AutoSnow (daily GeoTIFF)
    # ------------------------------------------------------
    df = collocate_AutoSnow(df, AutoSnow_meta, date_col="scan_date", out_col="AutoSnow")
    print("\nAVHRR + IMERG + ERA5 + MERRA2 + AutoSnow (sample rows):")
    print(df[["lon", "lat", "scan_line_times", "AutoSnow"]].head())
    _print_nan_diagnostics(df, "AutoSnow", "AutoSnow")

    # ------------------------------------------------------
    # 8) Save selected layers
    # ------------------------------------------------------
    print("\nSaving selected layers as GeoTIFF:")
    for layer in SAVE_LAYERS:
        if layer not in df.columns:
            print(f"WARNING: '{layer}' not found in df columns — skipping.")
            continue
        _save_layer_to_tiff(df, layer, avh_file, x_vec, y_vec)

    print("\n✅ AVHRR collocation test pipeline completed.\n")

    # ------------------------------------------------------
    # 9) Reproject selected WGS grids → Polar stereographic
    #     and save as NetCDF (Panoply-friendly)
    # ------------------------------------------------------
    DO_REPROJECT_TO_POLAR = True
    POLAR_OUT_DIR = "/home/omidzandi/test_maps_polar/"
    os.makedirs(POLAR_OUT_DIR, exist_ok=True)

    if DO_REPROJECT_TO_POLAR:
        print("\nReprojecting WGS grids to Polar stereographic (NH / SH)")

        orbit_tag = os.path.splitext(os.path.basename(avh_file))[0]

        # Build WGS grids once
        var_grids = {}
        for v in SAVE_LAYERS:
            if v not in df.columns:
                print(f"WARNING: '{v}' not found — skipping.")
                continue
            var_grids[v] = utils.df2grid(df, v, x_vec, y_vec).astype("float32")

        # Reproject all variables
        polar = reproject_vars_wgs_to_polar(
            var_grids,
            orbit_tag=orbit_tag,
            x_vec=x_vec,
            y_vec=y_vec,
            grid_resolution=GRID_RES,
            lat_thresh_nh=LAT_THRESH_NH,
            lat_thresh_sh=LAT_THRESH_SH,
            lat_ts_nh=70.0,
            lat_ts_sh=-71.0,
            nodata=-9999.0,
        )

        print("polar type:", type(polar))
        try:
            print("polar keys:", polar.keys())
        except Exception as e:
            print("polar has no keys():", e)

        # Save NH / SH NetCDFs
        utils.save_polar_netcdf(
            polar,
            out_dir=POLAR_OUT_DIR,
            orbit_tag=orbit_tag,
        )


if __name__ == "__main__":
    test_AVHRR_collocation()