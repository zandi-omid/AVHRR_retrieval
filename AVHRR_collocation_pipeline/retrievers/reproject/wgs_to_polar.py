"""
Reprojection utilities
----------------------

WGS84 (lat/lon) gridded arrays -> Polar stereographic (NH/SH) using GDAL warp.

- Writes intermediate GeoTIFFs into a TemporaryDirectory (fast, clean, no manual cleanup).
- Uses nearest-neighbor resampling (matches your previous workflow).
- Returns xarray.DataArray for single arrays, and dicts for multi-var workflows.

Author: Omid Zandi
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict, Tuple, Optional, Any

import numpy as np
import rasterio
import xarray as xr
from osgeo import gdal, osr


# Optional: reduce GDAL chatter
gdal.PushErrorHandler("CPLQuietErrorHandler")


def _save_geotiff_wgs(
    path: str,
    array2d: np.ndarray,
    *,
    minx: float,
    maxy: float,
    px_sz: float,
    nodata: float,
) -> None:
    """
    Save a (lat, lon) 2D array as a WGS84 GeoTIFF using GDAL.
    GeoTransform uses (minx, px_sz, 0, maxy, 0, -px_sz).
    """
    ypx, xpx = array2d.shape

    srs = osr.SpatialReference()
    srs.ImportFromProj4("+proj=longlat +datum=WGS84 +no_defs")
    proj_wkt = srs.ExportToWkt()

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        path,
        xpx,
        ypx,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW"],
    )

    ds.SetGeoTransform((float(minx), float(px_sz), 0.0, float(maxy), 0.0, -float(px_sz)))
    ds.SetProjection(proj_wkt)

    band = ds.GetRasterBand(1)
    band.SetNoDataValue(float(nodata))
    band.WriteArray(array2d.astype("float32", copy=False))
    band.FlushCache()
    ds.FlushCache()
    ds = None


def reproject_wgs_to_polar(
    arr: np.ndarray,
    hemisphere: str,
    lat_thresh: float,
    *,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    grid_resolution: float,
    lat_ts_nh: float = 70.0,
    lat_ts_sh: float = -71.0,
    nodata: float = -9999.0,
    tag: str = "reproj",
    tmp_root: Optional[str] = None,
) -> xr.DataArray:
    """
    Reproject a WGS84 gridded array to polar stereographic (NH or SH).

    Notes:
    - `arr` is assumed to be shaped (len(y_vec), len(x_vec)) i.e. (lat, lon).
    - Uses your original slicing logic:
        NH: y_vec > lat_thresh
        SH: y_vec <= lat_thresh
    - Writes temp GeoTIFFs inside a TemporaryDirectory and reads result back via rasterio.

    Parameters
    ----------
    arr : np.ndarray
        2D array on WGS grid (lat, lon).
    hemisphere : {"NH","SH"}
        Which hemisphere to produce.
    lat_thresh : float
        Threshold used for hemisphere slicing (same meaning as your old y_min).
    x_vec, y_vec : np.ndarray
        WGS grid vectors (centers).
    grid_resolution : float
        Grid spacing in degrees.
    lat_ts_nh, lat_ts_sh : float
        Latitude of true scale for polar stereographic.
    nodata : float
        Numeric nodata written to GeoTIFF (NaNs are converted to this before warp).
    tag : str
        Used to name temp files.
    tmp_root : str | None
        Root folder for TemporaryDirectory (defaults to $TMPDIR or /tmp).

    Returns
    -------
    xr.DataArray
        Reprojected polar stereographic array with coords x/y (meters) + crs attr.
    """
    if hemisphere not in ("NH", "SH"):
        raise ValueError("hemisphere must be 'NH' or 'SH'")

    # Slice according to your original convention
    if hemisphere == "NH":
        mask = y_vec > lat_thresh
        t_srs = f'+proj=stere +lat_0=90 +lat_ts={lat_ts_nh} +lon_0=0 +datum=WGS84'
    else:
        mask = y_vec <= lat_thresh
        t_srs = f'+proj=stere +lat_0=-90 +lat_ts={lat_ts_sh} +lon_0=0 +datum=WGS84'

    sliced = arr[mask, :]
    y_vec_sliced = y_vec[mask]

    # Replace NaNs with nodata so GDAL handles it reliably
    sliced = np.where(np.isfinite(sliced), sliced, nodata).astype("float32", copy=False)

    # Keep your convention (don’t overthink x_min/x_max)
    px_sz = float(grid_resolution)
    minx = float(np.min(x_vec) + grid_resolution / 2.0)
    maxy = float(np.max(y_vec_sliced) - grid_resolution / 2.0)

    if tmp_root is None:
        tmp_root = os.environ.get("TMPDIR", "/tmp")

    env = os.environ.copy()
    env.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
        wgs_tif = os.path.join(tmp_dir, f"{tag}_{hemisphere.lower()}_wgs.tif")
        stereo_tif = os.path.join(tmp_dir, f"{tag}_{hemisphere.lower()}_stereo.tif")

        _save_geotiff_wgs(
            wgs_tif,
            sliced,
            minx=minx,
            maxy=maxy,
            px_sz=px_sz,
            nodata=nodata,
        )

        cmd = (
            "gdalwarp -overwrite "
            f"-srcnodata {nodata} -dstnodata {nodata} "
            f"-t_srs \"{t_srs}\" "
            "-r near "
            "-co COMPRESS=LZW "
            f"\"{wgs_tif}\" \"{stereo_tif}\""
        )
        p = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
        if p.returncode != 0:
            raise RuntimeError(f"gdalwarp failed (return code {p.returncode}). Command:\n{cmd}")

        with rasterio.open(stereo_tif) as src:
            data = src.read(1).astype("float32", copy=False)
            data = np.where(data == nodata, np.nan, data)

            tr = src.transform
            xs = tr.c + tr.a * np.arange(src.width)
            ys = tr.f + tr.e * np.arange(src.height)

            da = xr.DataArray(
                data,
                coords={"y": ys, "x": xs},
                dims=("y", "x"),
                attrs={"crs": src.crs.to_string()},
            )

    return da


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
    nodata: float = -9999.0,
    tmp_root: Optional[str] = None,
    return_coords: bool = False,
) -> Dict[str, Any]:
    """
    Reproject a dict of WGS grids -> polar stereographic for NH and SH.

    Parameters
    ----------
    var_grids : dict[str, np.ndarray]
        Each array must be on the same WGS grid (lat, lon) matching y_vec/x_vec.
    orbit_tag : str
        Used in temp file naming.
    lat_thresh_nh, lat_thresh_sh : float
        Hemisphere thresholds (your original y_min logic).
    return_coords : bool
        If True, also return x/y coord vectors for NH/SH.

    Returns
    -------
    dict
        Always returns:
          {
            "NH": {var: (("y","x"), array2d), ...},
            "SH": {var: (("y","x"), array2d), ...},
          }

        If return_coords=True, also includes:
          {
            "coords": {
              "NH": {"x": x_coords_nh, "y": y_coords_nh},
              "SH": {"x": x_coords_sh, "y": y_coords_sh},
            }
          }
    """
    vars_nh: Dict[str, Tuple[Tuple[str, str], np.ndarray]] = {}
    vars_sh: Dict[str, Tuple[Tuple[str, str], np.ndarray]] = {}

    x_coords_nh = y_coords_nh = None
    x_coords_sh = y_coords_sh = None

    for vname, grid in var_grids.items():
        grid = grid.astype("float32", copy=False)

        da_nh = reproject_wgs_to_polar(
            grid,
            "NH",
            lat_thresh_nh,
            x_vec=x_vec,
            y_vec=y_vec,
            grid_resolution=grid_resolution,
            lat_ts_nh=lat_ts_nh,
            lat_ts_sh=lat_ts_sh,
            nodata=nodata,
            tag=f"{orbit_tag}_{vname}",
            tmp_root=tmp_root,
        )
        vars_nh[vname] = (("y", "x"), da_nh.values.astype("float32", copy=False))
        if x_coords_nh is None:
            x_coords_nh = da_nh["x"].values
            y_coords_nh = da_nh["y"].values

        da_sh = reproject_wgs_to_polar(
            grid,
            "SH",
            lat_thresh_sh,
            x_vec=x_vec,
            y_vec=y_vec,
            grid_resolution=grid_resolution,
            lat_ts_nh=lat_ts_nh,
            lat_ts_sh=lat_ts_sh,
            nodata=nodata,
            tag=f"{orbit_tag}_{vname}",
            tmp_root=tmp_root,
        )
        vars_sh[vname] = (("y", "x"), da_sh.values.astype("float32", copy=False))
        if x_coords_sh is None:
            x_coords_sh = da_sh["x"].values
            y_coords_sh = da_sh["y"].values

    out: Dict[str, Any] = {"NH": vars_nh, "SH": vars_sh}

    if return_coords:
        out["coords"] = {
            "NH": {"x": x_coords_nh, "y": y_coords_nh},
            "SH": {"x": x_coords_sh, "y": y_coords_sh},
        }

    return out