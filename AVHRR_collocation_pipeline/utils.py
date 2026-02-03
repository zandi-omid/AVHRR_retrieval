# import required packages
import numpy as np
import os 
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from typing import Tuple
import random
from typing import Union, List, Dict, Any, Optional
import xarray as xr
import datetime



# functions

def index_finder(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_bins: np.ndarray,
    lat_bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map input longitude/latitude values to grid-cell indices
    defined by target longitude and latitude bin edges/vectors.

    Parameters
    ----------
    lon : np.ndarray
        1D or flattened array of longitudes (any order).
    lat : np.ndarray
        1D or flattened array of latitudes (any order).
    lon_bins : np.ndarray
        1D array of grid longitude centers *in ascending order*
        (e.g., -180, -179.5, ...).
    lat_bins : np.ndarray
        1D array of grid latitude centers *in descending order*
        (e.g., 90, 89.5, ...).

    Returns
    -------
    idx : np.ndarray
        X-index for each lon element (−1 where outside grid).
    idy : np.ndarray
        Y-index for each lat element (−1 where outside grid).

    Notes
    -----
    - Uses np.digitize() for fast binning.
    - Any point outside the grid returns index -1.
    - This function is consistent with the AVHRR reader and df2grid.
    """

    # Ensure numpy arrays
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # Digitize longitudes
    idx = np.digitize(lon, lon_bins) - 1

    # Latitude digitization requires right=True because bins decrease
    idy = np.digitize(lat, lat_bins, right=True) - 1

    # Mask out-of-range values
    idx[(idx < 0) | (idx >= len(lon_bins))] = -1
    idy[(idy < 0) | (idy >= len(lat_bins))] = -1

    return idx, idy


def df2grid(
    df,
    var_name: str,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Old-style df2grid, matching Omid's original pipeline.

    - Uses index_finder(lon, lat, x_vec, y_vec)
    - Writes directly into a grid shaped like x,y
    - Keeps NaNs as in the original implementation
    """
    # Compute indices using the SAME binning logic as before
    idx, idy = index_finder(df["lon"].values, df["lat"].values, x_vec, y_vec)

    # Optional safety: only keep points that land on the grid
    valid = (idx >= 0) & (idy >= 0) & np.isfinite(df[var_name].values)

    grid = np.full_like(x, np.nan, dtype=float)

    grid[idy[valid], idx[valid]] = df[var_name].values[valid]

    return grid

def grid_swath_var_mean(
    swath_ds: xr.Dataset,
    var_name: str,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    *,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> np.ndarray:
    """
    Grid a swath variable (on the AVHRR swath grid) to a regular WGS grid (y,x)
    using mean aggregation per grid cell.

    This is consistent with the normal of using utils.index_finder()
    (x_vec ascending, y_vec descending).

    Parameters
    ----------
    swath_ds : xr.Dataset
        Dataset containing:
          - lat_name: 2D latitude (lines, pixels)
          - lon_name: 2D longitude (lines, pixels)
          - var_name: 2D variable on same swath grid
    var_name : str
        Variable to grid (e.g., "temp_11_0um_nom_corrected").
    x_vec, y_vec : np.ndarray
        Target WGS grid vectors (lon centers ascending, lat centers descending).
    lat_name, lon_name : str
        Names of latitude/longitude variables in swath_ds.

    Returns
    -------
    grid : np.ndarray
        2D array with shape (len(y_vec), len(x_vec)) containing mean values.
        Cells with no samples are NaN.
    """
    if lat_name not in swath_ds:
        raise KeyError(f"swath_ds missing '{lat_name}'")
    if lon_name not in swath_ds:
        raise KeyError(f"swath_ds missing '{lon_name}'")
    if var_name not in swath_ds:
        raise KeyError(f"swath_ds missing '{var_name}'")

    lat = np.asarray(swath_ds[lat_name].values)
    lon = np.asarray(swath_ds[lon_name].values)
    val = np.asarray(swath_ds[var_name].values)

    # flatten
    lat_f = lat.ravel()
    lon_f = lon.ravel()
    val_f = val.ravel()

    # keep only finite samples
    ok = np.isfinite(lat_f) & np.isfinite(lon_f) & np.isfinite(val_f)
    if not np.any(ok):
        # return all-NaN grid
        return np.full((len(y_vec), len(x_vec)), np.nan, dtype=np.float32)

    lat_f = lat_f[ok]
    lon_f = lon_f[ok]
    val_f = val_f[ok].astype(np.float64)

    # map to grid indices using YOUR binning rules
    ix, iy = index_finder(lon_f, lat_f, x_vec, y_vec)
    on = (ix >= 0) & (iy >= 0)

    if not np.any(on):
        return np.full((len(y_vec), len(x_vec)), np.nan, dtype=np.float32)

    ix = ix[on].astype(np.int64)
    iy = iy[on].astype(np.int64)
    val_f = val_f[on]

    nx = len(x_vec)
    ny = len(y_vec)

    # flatten cell index
    cell = iy * nx + ix
    ncell = nx * ny

    # sum + count per cell
    s = np.bincount(cell, weights=val_f, minlength=ncell)
    c = np.bincount(cell, minlength=ncell)

    out = np.full(ncell, np.nan, dtype=np.float32)
    m = np.divide(s, c, out=np.zeros_like(s), where=(c > 0))
    out[c > 0] = m[c > 0].astype(np.float32)

    return out.reshape((ny, nx))


def save_grid_to_tiff(
    grid: np.ndarray,
    outfile: str,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    nodata: float = np.nan,
) -> None:
    """
    Save a 2D numpy grid (lat/lon aligned) to a GeoTIFF file.

    Parameters
    ----------
    grid : np.ndarray
        2D array (lat-major) of values.
    outfile : str
        Path to output .tif file.
    x_vec : np.ndarray
        1D longitude centers (ascending).
    y_vec : np.ndarray
        1D latitude centers (descending).
    nodata : float, optional
        Value to write for missing data (default: NaN).

    Notes
    -----
    - CRS = WGS84 (EPSG:4326)
    - Uses regular-grid transform derived from x_vec/y_vec.
    """

    # -------------------------
    # Validate array dimensions
    # -------------------------
    if grid.ndim != 2:
        raise ValueError("Grid must be a 2D numpy array")

    if len(x_vec) != grid.shape[1]:
        raise ValueError(f"x_vec length {len(x_vec)} does not match grid width {grid.shape[1]}")

    if len(y_vec) != grid.shape[0]:
        raise ValueError(f"y_vec length {len(y_vec)} does not match grid height {grid.shape[0]}")

    # -------------------------
    # Build raster transform
    # -------------------------
    resolution_x = x_vec[1] - x_vec[0]
    resolution_y = y_vec[0] - y_vec[1]

    transform = from_origin(
        west=x_vec.min(),
        north=y_vec.max(),
        xsize=resolution_x,
        ysize=abs(resolution_y),
    )

    # -------------------------
    # Ensure directory exists
    # -------------------------
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # -------------------------
    # Write GeoTIFF
    # -------------------------
    metadata = {
        "driver": "GTiff",
        "height": grid.shape[0],
        "width": grid.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": nodata,
    }

    grid_to_write = grid.astype(np.float32)

    with rasterio.open(outfile, "w", **metadata) as dst:
        dst.write(grid_to_write, 1)


def pick_random_nc_file(folders: Union[str, List[str]]) -> str:
    """
    Return a random NetCDF (.nc) file from one or more directories.

    Parameters
    ----------
    folders : str or list[str]
        One folder (as a string) or multiple folders (as list of strings).

    Returns
    -------
    str
        Path to a randomly selected .nc file.

    Raises
    ------
    FileNotFoundError
        If no .nc files exist in any provided directory.
    TypeError
        If folders is not a string or list of strings.
    """

    # Normalize input to list
    if isinstance(folders, str):
        folder_list = [folders]
    elif isinstance(folders, list):
        folder_list = folders
    else:
        raise TypeError(
            "folders must be a string or a list of strings, "
            f"received type={type(folders).__name__}"
        )

    all_files: List[str] = []

    for folder in folder_list:
        if not os.path.isdir(folder):
            print(f"WARNING: Directory does not exist → {folder}")
            continue

        nc_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".nc")
        ]

        if len(nc_files) == 0:
            print(f"WARNING: No .nc files found in → {folder}")

        all_files.extend(nc_files)

    if len(all_files) == 0:
        raise FileNotFoundError(
            "No .nc files found in the provided folder(s): "
            f"{folder_list}"
        )

    return random.choice(all_files)


def build_test_grid(resolution: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a global lat/lon grid for testing.

    Parameters
    ----------
    resolution : float, optional
        Grid spacing in degrees. Default is 0.5°.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        x_vec : 1D array of longitudes
        y_vec : 1D array of latitudes
        x : 2D longitude grid
        y : 2D latitude grid
    """
    x_vec = np.arange(-180, 180, resolution)
    y_vec = np.arange(90, -90, -resolution)
    x, y = np.meshgrid(x_vec, y_vec)
    return x_vec, y_vec, x, y


def build_tiff_name(
    avh_file: str,
    var_name: str,
    out_dir: str,
) -> str:
    """
    Build a standardized output filename for saving a gridded variable
    from an AVHRR orbit or the collocated dataset as a GeoTIFF file.

    The naming convention is:
        <orbit_basename>__<variable_name>.tif

    Parameters
    ----------
    avh_file : str
        Path to the original AVHRR orbit NetCDF file.
    var_name : str
        Name of the variable being exported (e.g., "temp_12_0um_nom").
    out_dir : str
        Directory where the TIFF file will be saved.

    Returns
    -------
    str
        Full path to the output TIFF file.

    Notes
    -----
    - The `.nc` extension of the input file is removed.
    """

    base = os.path.basename(avh_file)          # "clavrx_NSS....nc"
    base_no_ext = os.path.splitext(base)[0]    # "clavrx_NSS...."

    fname = f"{var_name}_{base_no_ext}.tif"   # standardized name

    return os.path.join(out_dir, fname)


def add_time_columns(
    df: pd.DataFrame,
    time_col: str = "scan_line_times",
    *,
    add_nearest_hour: bool = True,
    add_nearest_halfhour: bool = True,
    round_ties: str = "half_even",  # choose "half_even" to better mimic pandas dt.round("h")
) -> pd.DataFrame:
    """
    Add standardized time columns to df once, for use by ALL collocators.

    Requires:
      df[time_col] = UNIX seconds (int or float). Floats can occur after aggregation.

    Adds (general):
      - scan_unix             float64 original unix seconds (preserves fractions)
      - scan_dt               datetime64[s] (floor to second, for display)
      - scan_date             str "YYYY-MM-DD" (from scan_dt)
      - scan_hour             int16 0..23 (from scan_dt)
      - scan_hour_unix        int64 unix seconds of nearest hour (optional)
      - scan_date_hr          str date from scan_hour_unix (optional)
      - scan_hour_hr          int16 hour from scan_hour_unix (optional)
      - scan_halfhour_unix    int64 unix seconds of nearest 30-min (optional)

    Adds (OLD MERRA2-matching keys):
      - scan_dt_round_h       datetime64[s] timestamp rounded to nearest hour
                              (ties handled by round_ties; "half_even" mimics pandas better)
      - scan_date_m2          str "YYYY-MM-DD" derived from scan_dt_round_h
      - scan_hour_m2          int16 hour derived from scan_dt_round_h

    Notes
    -----
    Old pipeline used:
        scan_line_times_round = pd.to_datetime(scan_line_times, unit="s").dt.round("h")
        scan_line_hours_int_no_1_hr = scan_line_times_round.dt.hour
        scan_line_y-m-d = scan_line_times_round.strftime("%Y-%m-%d")

    To match that as closely as possible WITHOUT pandas, use:
        round_ties="half_even"  (banker's rounding at exact .5 hour)
    """
    out = df.copy()

    # Keep float to avoid truncation artifacts after groupby mean
    t = out[time_col].to_numpy(dtype="float64")
    out["scan_unix"] = t

    # Display datetime (floored to second)
    t_sec_floor = np.floor(t).astype("int64")
    scan_dt = t_sec_floor.astype("datetime64[s]")
    day = scan_dt.astype("datetime64[D]")

    out["scan_dt"] = scan_dt
    out["scan_date"] = day.astype(str)

    hour = (scan_dt.astype("datetime64[h]") - day).astype("timedelta64[h]").astype(int)
    out["scan_hour"] = hour.astype(np.int16)

    # ------------------------------------------------------------
    # OLD MERRA2-style: round to nearest hour (EXACT, float-safe)
    # # Match old pipeline: drop fractional seconds, don't round them
    # ------------------------------------------------------------
    t_sec = t_sec_floor  # snap to whole seconds first
    sec_in_hour = 3600

    rem = t_sec % sec_in_hour           # seconds past the hour
    down = t_sec - rem                  # lower hour boundary (unix sec)
    up = down + sec_in_hour             # upper hour boundary

    # nearest-hour decision
    choose_up = rem > (sec_in_hour // 2)        # > 1800
    choose_down = rem < (sec_in_hour // 2)      # < 1800
    tie = rem == (sec_in_hour // 2)             # exactly 1800 sec

    if round_ties == "half_up":
        hr_unix_round = np.where(choose_up | tie, up, down).astype("int64")
    elif round_ties == "half_even":
        # banker's rounding: ties go to EVEN hour index
        down_hour_index = (down // sec_in_hour).astype("int64")
        even_down = (down_hour_index % 2 == 0)
        tie_choose_up = tie & (~even_down)
        hr_unix_round = np.where(choose_up | tie_choose_up, up, down).astype("int64")
    else:
        raise ValueError("round_ties must be 'half_up' or 'half_even'")
    
    # Old-style rounded datetime and keys
    dt_round_h = hr_unix_round.astype("datetime64[s]")
    day_round = dt_round_h.astype("datetime64[D]")

    out["scan_dt_round_h"] = dt_round_h
    out["scan_date_m2"] = day_round.astype(str)
    out["scan_hour_m2"] = (
        (dt_round_h.astype("datetime64[h]") - day_round)
        .astype("timedelta64[h]")
        .astype(int)
        .astype(np.int16)
    )

    # ------------------------------------------------------------
    # Nearest-hour unix (keep your existing outputs, but use the SAME rounding)
    # so scan_date_hr/scan_hour_hr remain consistent and reproducible.
    # ------------------------------------------------------------
    if add_nearest_hour:
        out["scan_hour_unix"] = hr_unix_round

        dt_hr = dt_round_h
        day_hr = day_round

        out["scan_date_hr"] = day_hr.astype(str)
        out["scan_hour_hr"] = (
            (dt_hr.astype("datetime64[h]") - day_hr)
            .astype("timedelta64[h]")
            .astype(int)
            .astype(np.int16)
        )

        out["scan_hour_unix_era5"] = (hr_unix_round + 3600).astype("int64")

    # ------------------------------------------------------------
    # Nearest half-hour unix
    # ------------------------------------------------------------
    if add_nearest_halfhour:
        hh_unix = (np.floor((t + 900.0) / 1800.0) * 1800.0).astype("int64")
        out["scan_halfhour_unix"] = hh_unix

    return out


def save_polar_netcdf(
    polar_vars: Dict[str, Any],
    *,
    orbit_tag: str,
    out_dir: str,
    compress_level: int = 4,
) -> None:
    """
    Save NH / SH polar-stereographic variables to NetCDF.

    Expected structure (minimum):
        polar_vars = {
            "NH": {varname: (("y","x"), array2d), ...},
            "SH": {varname: (("y","x"), array2d), ...},
        }

    Optional (if you called reproject_vars_wgs_to_polar(..., return_coords=True)):
        polar_vars["coords"] = {
            "NH": {"x": x_coords_nh, "y": y_coords_nh},
            "SH": {"x": x_coords_sh, "y": y_coords_sh},
        }
    """
    if not isinstance(polar_vars, dict):
        raise TypeError("polar_vars must be a dict")

    os.makedirs(out_dir, exist_ok=True)

    coords_block: Optional[Dict[str, Any]] = None
    if "coords" in polar_vars:
        if isinstance(polar_vars["coords"], dict):
            coords_block = polar_vars["coords"]

    for hemi in ("NH", "SH"):
        hemi_vars = polar_vars.get(hemi, None)
        if hemi_vars is None:
            print(f"[WARN] {hemi} not found in polar_vars — skipping")
            continue
        if not isinstance(hemi_vars, dict):
            raise TypeError(f"polar_vars['{hemi}'] must be a dict of variables")

        ds = xr.Dataset(hemi_vars)

        # Attach coords (strongly recommended for Panoply)
        if coords_block is not None and hemi in coords_block:
            c = coords_block[hemi]
            x = c.get("x", None)
            y = c.get("y", None)

            if x is not None and y is not None:
                ds = ds.assign_coords(
                    x=np.asarray(x, dtype="float64"),
                    y=np.asarray(y, dtype="float64"),
                )

        out_nc = os.path.join(out_dir, f"{orbit_tag}_{hemi}_polar.nc")

        encoding = {v: {"zlib": True, "complevel": int(compress_level)} for v in ds.data_vars}

        ds.to_netcdf(out_nc, format="NETCDF4", encoding=encoding)
        print(f"Saved {hemi} polar NetCDF → {out_nc}")

def AVHRR_datetime_NC_files(AVHRR_file):

    name_splited = os.path.basename(AVHRR_file).split('.')

    yr_DOY = name_splited[3][1:]
    st_time = name_splited[4][1:]
    end_time = name_splited[5][1:]
    yr = datetime.datetime.strptime(yr_DOY, '%y%j').year

    st_dt = datetime.datetime.strptime(yr_DOY + st_time, '%y%j%H%M')

    end_dt = datetime.datetime.strptime(yr_DOY + end_time, '%y%j%H%M')

    if end_dt < st_dt:
        #files which ending time is few moments after 00:00 pm
        end_dt = end_dt + datetime.timedelta(days=1)

    return st_dt, end_dt, yr