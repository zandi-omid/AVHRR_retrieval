import os

from AVHRR_collocation_pipeline.readers.IMERG_reference import load_IMERG_reference
from AVHRR_collocation_pipeline.readers.ERA5_reference import load_ERA5_reference
from AVHRR_collocation_pipeline.readers.AutoSnow_reference import load_AutoSnow_reference
from AVHRR_collocation_pipeline.readers.MERRA2_reference import load_MERRA2_reference

from AVHRR_collocation_pipeline.retrievers.collocate_and_reproj import AVHRRProcessor

import AVHRR_collocation_pipeline.utils as utils


# -----------------------------------------
# CONFIG
# -----------------------------------------
AVHRR_FOLDERS = [
    "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AVHRR/2019",
]

OUT_DIR = "/xdisk/behrangi/omidzandi/retrieved_maps/test/"
os.makedirs(OUT_DIR, exist_ok=True)

GRID_RES: float = 0.5  # degrees

# Hemisphere split thresholds (also reusable later for reprojection)
LAT_THRESH_NH: float = 55.0
LAT_THRESH_SH: float = -55.0


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
    "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/MERRA2/merra2_archive_19800101_20250831"
)

# Which MERRA2 variables to collocate (must exist inside your daily files)
MERRA2_VARS: list[str] = [
    "TQV",
    "T2M",
]

AUTOSNOW_DIR: str = (
    "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AutoSnow/autosnow_in_geotif"
)


# Polar reprojection controls
DO_REPROJECT_TO_POLAR: bool = True
POLAR_OUT_DIR: str = "/xdisk/behrangi/omidzandi/"
os.makedirs(POLAR_OUT_DIR, exist_ok=True)

LAT_TS_NH: float = 70.0
LAT_TS_SH: float = -71.0

# -----------------------------------------
# LOAD REFERENCES ONCE FOR THE TEST
# -----------------------------------------
# IMERG_meta = load_IMERG_reference(IMERG_DIR)
# ERA5_meta_by_year = load_ERA5_reference(ERA5_DIR)
MERRA2_meta = load_MERRA2_reference(MERRA2_DIR)
AutoSnow_meta = load_AutoSnow_reference(AUTOSNOW_DIR)

def test_processor() -> None:
    """
    Test the AVHRRProcessor class:
      - pick a random AVHRR orbit
      - collocate IMERG/ERA5/MERRA2/AutoSnow
      - grid & reproject to polar
      - save polar NetCDFs in /home/omidzandi/test_maps_polar
    """

    # 1) Pick a random AVHRR file
    avh_file: str = utils.pick_random_nc_file(AVHRR_FOLDERS)
    print(f"\nSelected AVHRR orbit:\n  {avh_file}\n")

    # 2) Build the processor
    processor = AVHRRProcessor(
        grid_res=GRID_RES,
        lat_thresh_nh=LAT_THRESH_NH,
        lat_thresh_sh=LAT_THRESH_SH,
        lat_ts_nh=LAT_TS_NH,
        lat_ts_sh=LAT_TS_SH,
        nodata=-9999.0,
        imerg_meta=None,
        era5_meta_by_year=None,
        merra2_meta=MERRA2_meta,
        autosnow_meta=AutoSnow_meta,
    )

    # 3) Define DL inputs & MERRA2 vars
    input_vars = [
        "cloud_probability",
        "temp_11_0um_nom",
        "temp_12_0um_nom",
        "TQV",
        "T2M",
        "AutoSnow",
    ]

    # 4) Run process_orbit with your desired config
    polar = processor.process_orbit(
        avh_file=avh_file,
        avh_vars=["cloud_probability", "temp_11_0um_nom", "temp_12_0um_nom"],
        input_vars=input_vars,
        merra2_vars=["TQV", "T2M"],
        need_imerg=False,
        need_era5=False,
        extra_eval_vars=["IMERG_preci", "ERA5_tp"],
        save_polar_dir="/xdisk/behrangi/omidzandi/retrieved_maps",
    )

    print("\n✅ Processor test finished.")
    print("Type of polar:", type(polar))
    if isinstance(polar, dict):
        print("polar keys:", polar.keys())
        for hemi in ("NH", "SH"):
            if hemi in polar:
                print(f"  {hemi}: vars = {list(polar[hemi].keys())}")

if __name__ == "__main__":
    test_processor()