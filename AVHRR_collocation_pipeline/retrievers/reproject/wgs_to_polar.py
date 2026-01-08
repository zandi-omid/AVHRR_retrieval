# AVHRR_collocation_pipeline/retrievers/reproject/wgs_to_polar.py

import os
import tempfile
import subprocess
from typing import Optional

import numpy as np
import xarray as xr
from osgeo import gdal, osr
import rasterio


def _gdal_based_save_array_to_disk(
    dst_filename: str,
    xpx: int,
    ypx: int,
    px_sz: float,
    minx: float,
    maxy: float,
    crs: str,
    crs_format: str,
    array_to_save: np.ndarray,
) -> None:
    """
    Exact port of your old gdal_based_save_array_to_disk.

    Saves a 2D array to GeoTIFF with given geotransform and CRS.
    Clamps bottom latitude to -90 if necessary (matching old behavior).
    """
    # Clamp bottom latitude
    bottom_latitude = maxy - (ypx * px_sz)
    if bottom_latitude < -90.0:
        bottom_latitude = -90.0
        ypx = int((maxy - bottom_latitude) / px_sz)
        array_to_save = array_to_save[:ypx, :]

    srs = osr.SpatialReference()
    if crs_format == "proj4":
        srs.ImportFromProj4(crs)
        srs = srs.ExportToProj4()
    elif crs_format == "WKT":
        srs.ImportFromWkt(crs)
        srs = srs.ExportToWkt()

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(dst_filename, xpx, ypx, 1, gdal.GDT_Float64)

    ds.SetGeoTransform(
        (
            minx,   # top-left x
            px_sz,  # pixel width
            0.0,    # rotation
            maxy,   # top-left y
            0.0,    # rotation
            -px_sz, # pixel height (negative)
        )
    )

    ds.SetProjection(srs)
    band = ds.GetRasterBand(1)
    band.WriteArray(array_to_save)
    ds.FlushCache()
    del ds


def reproject_wgs_to_polar(
    grid: np.ndarray,
    hemisphere: str,
    lat_thresh: float,
    *,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    grid_resolution: float,
    lat_ts_nh: float = 70.0,
    lat_ts_sh: float = -71.0,
    tag: str = "reproj",
    tmp_root: Optional[str] = None,
) -> xr.DataArray:
    """
    WGS84 → polar stereographic reprojection that reproduces the old
    gdal_based_reproj_arr behavior as closely as possible.

    Parameters
    ----------
    grid : 2D np.ndarray
        Global WGS84 grid (lat along axis 0, lon along axis 1).
    hemisphere : {"NH", "SH"}
        Hemisphere to slice and reproject.
    lat_thresh : float
        Latitude threshold used for NH/SH split (your y_min in old code).
    x_vec, y_vec : np.ndarray
        Longitude / latitude vectors for the global grid.
    grid_resolution : float
        Grid spacing in degrees (e.g., 0.25).
    lat_ts_nh, lat_ts_sh : float
        Standard parallels for stereographic projection (unchanged from old).
    tag : str
        Name for the returned DataArray.
    tmp_root : str or None
        Directory to write temporary GeoTIFFs; if None, uses system temp.
    """

    # --- 1) Slice grid in latitude exactly like the old script ---

    if hemisphere == "NH":
        # y_min in old code == lat_thresh
        sliced_arr = grid[y_vec > lat_thresh]
    else:
        sliced_arr = grid[y_vec <= lat_thresh]

    ypx, xpx = sliced_arr.shape
    px_sz = float(grid_resolution)

    # This is exactly what you had in gdal_based_reproj_arr
    minx = float(np.min(x_vec) + grid_resolution / 2.0)
    if hemisphere == "NH":
        maxy = float(np.max(y_vec) - grid_resolution / 2.0)
    else:  # SH
        maxy = float(lat_thresh - grid_resolution / 2.0)

    # --- 2) Save hemisphere WGS GeoTIFF using the same CRS and logic as before ---

    crs = "+proj=longlat +datum=WGS84 +no_defs"
    crs_format = "proj4"

    tmp_dir = tmp_root or tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)

    base = f"tmp_{hemisphere}_{os.getpid()}"
    wgs_tif = os.path.join(tmp_dir, base + "_wgs.tif")
    stereo_tif = os.path.join(tmp_dir, base + "_stereo.tif")

    _gdal_based_save_array_to_disk(
        wgs_tif,
        xpx=xpx,
        ypx=ypx,
        px_sz=px_sz,
        minx=minx,
        maxy=maxy,
        crs=crs,
        crs_format=crs_format,
        array_to_save=sliced_arr,
    )

    # --- 3) gdalwarp with the SAME options as the old script ---

    if hemisphere == "NH":
        t_srs = f'+proj=stere +lat_0=90 +lat_ts={lat_ts_nh} +lon_0=0 +datum=WGS84'
    else:
        t_srs = f'+proj=stere +lat_0=-90 +lat_ts={lat_ts_sh} +lon_0=0 +datum=WGS84'

    gdalwarp_cmd = (
        'gdalwarp -dstnodata nan -wo NUM_THREADS=1 '
        f'-t_srs "{t_srs}" '
        '-r near '
        f'{wgs_tif} {stereo_tif}'
    )
    subprocess.run(gdalwarp_cmd, shell=True, stdout=subprocess.DEVNULL, check=True)

    # --- 4) Read polar raster and build xarray.DataArray with x/y coords ---

    with rasterio.open(stereo_tif) as src:
        data = src.read(1).astype("float32")
        transform = src.transform

        dx = transform.a
        dy = transform.e
        x0 = transform.c
        y0 = transform.f

        height, width = data.shape
        x = x0 + dx * np.arange(width)
        y = y0 + dy * np.arange(height)

        da = xr.DataArray(
            data,
            coords={
                "y": y.astype("float64"),
                "x": x.astype("float64"),
            },
            dims=("y", "x"),
            name=tag,
            attrs={"crs": src.crs.to_string() if src.crs is not None else ""},
        )

    # --- 5) Clean up temp files ---

    try:
        os.remove(wgs_tif)
    except OSError:
        pass
    try:
        os.remove(stereo_tif)
    except OSError:
        pass

    return da

from typing import Dict, Tuple

def reproject_vars_wgs_to_polar(
    var_grids: Dict[str, np.ndarray],
    *,
    orbit_tag: str,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    grid_resolution: float,
    lat_thresh_nh: float,
    lat_thresh_sh: float,
    lat_ts_nh: float = 70.0,
    lat_ts_sh: float = -71.0,
    nodata: float = -9999.0,   # kept for API compatibility (not used internally)
    tmp_root: Optional[str] = None,
) -> Dict:
    """
    Wrapper used by AVHRRProcessor.wgs_to_polar.

    Returns a dict with:
      polar["NH"]      -> {var: (("y","x"), arr_nh)}
      polar["SH"]      -> {var: (("y","x"), arr_sh)}
      polar["coords"]  -> {"NH": {"x": x_nh, "y": y_nh},
                           "SH": {"x": x_sh, "y": y_sh}}
    matching expectations in write_polar_groups_netcdf.
    """

    nh_vars: Dict[str, Tuple[Tuple[str, str], np.ndarray]] = {}
    sh_vars: Dict[str, Tuple[Tuple[str, str], np.ndarray]] = {}

    nh_x = nh_y = None
    sh_x = sh_y = None

    for vname, grid in var_grids.items():
        # NH
        da_nh = reproject_wgs_to_polar(
            grid,
            "NH",
            lat_thresh_nh,
            x_vec=x_vec,
            y_vec=y_vec,
            grid_resolution=grid_resolution,
            lat_ts_nh=lat_ts_nh,
            lat_ts_sh=lat_ts_sh,
            tag=vname,
            tmp_root=tmp_root,
        )

        # SH
        da_sh = reproject_wgs_to_polar(
            grid,
            "SH",
            lat_thresh_sh,
            x_vec=x_vec,
            y_vec=y_vec,
            grid_resolution=grid_resolution,
            lat_ts_nh=lat_ts_nh,
            lat_ts_sh=lat_ts_sh,
            tag=vname,
            tmp_root=tmp_root,
        )

        nh_vars[vname] = (("y", "x"), da_nh.values.astype("float32"))
        sh_vars[vname] = (("y", "x"), da_sh.values.astype("float32"))

        if nh_x is None:
            nh_x = da_nh["x"].values.astype("float64")
            nh_y = da_nh["y"].values.astype("float64")
        if sh_x is None:
            sh_x = da_sh["x"].values.astype("float64")
            sh_y = da_sh["y"].values.astype("float64")

    polar = {
        "NH": nh_vars,
        "SH": sh_vars,
        "coords": {
            "NH": {"x": nh_x, "y": nh_y},
            "SH": {"x": sh_x, "y": sh_y},
        },
    }

    return polar