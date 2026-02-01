"""
Polar stereographic -> WGS84 reprojection utilities
--------------------------------------------------

Takes polar stereo arrays with x/y coordinate vectors and reprojects to EPSG:4326
using GDAL warp. Uses TemporaryDirectory to avoid manual cleanup.

Author: Omid Zandi
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict, Optional, Any

import numpy as np
import rasterio
import xarray as xr


def _affine_from_xy_vectors(x_vec: np.ndarray, y_vec: np.ndarray) -> rasterio.Affine:
    """
    Build an affine transform from 1D x/y coordinate vectors (cell centers).

    Assumes regular spacing, and y decreasing (typical for raster grids).
    """
    if x_vec.size < 2 or y_vec.size < 2:
        raise ValueError("x_vec and y_vec must have at least 2 elements.")

    dx = float(x_vec[1] - x_vec[0])
    dy = float(y_vec[1] - y_vec[0])  # often negative

    # Convert from centers -> top-left origin convention
    x0 = float(np.min(x_vec) - dx / 2.0)
    y0 = float(np.max(y_vec) + dy / 2.0)  # dy < 0 typical

    return rasterio.Affine(dx, 0.0, x0, 0.0, dy, y0)


def polar_crs_proj4(
    hemisphere: str,
    *,
    lat_ts_nh: float = 70.0,
    lat_ts_sh: float = -71.0,
) -> str:
    """
    Build proj4 string for polar stereographic.
    """
    if hemisphere not in ("NH", "SH"):
        raise ValueError("hemisphere must be 'NH' or 'SH'")

    if hemisphere == "NH":
        return f"+proj=stere +lat_0=90 +lat_ts={lat_ts_nh} +lon_0=0 +datum=WGS84"
    return f"+proj=stere +lat_0=-90 +lat_ts={lat_ts_sh} +lon_0=0 +datum=WGS84"


def reproject_polar_to_wgs(
    arr: np.ndarray,
    *,
    hemisphere: str,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    grid_resolution_deg: float,
    lat_ts_nh: float = 70.0,
    lat_ts_sh: float = -71.0,
    resampling: str = "near",
    nodata: float = np.nan,
    tag: str = "polar2wgs",
    tmp_root: Optional[str] = None,
) -> xr.DataArray:
    """
    Reproject a single polar stereo 2D array -> WGS84 (EPSG:4326).

    Parameters
    ----------
    arr : np.ndarray
        2D array in polar stereo grid (y, x)
    hemisphere : {"NH","SH"}
    x_vec, y_vec : np.ndarray
        1D coordinate vectors in meters (cell centers)
    grid_resolution_deg : float
        Output resolution (deg), e.g. 0.25 or 0.1
    resampling : str
        gdalwarp resampling: near, bilinear, cubic, etc.
    nodata : float
        nodata marker; if np.nan, we pass "nan" to gdalwarp (works in your workflow)
    """
    if tmp_root is None:
        tmp_root = os.environ.get("TMPDIR", "/tmp")

    s_srs = polar_crs_proj4(hemisphere, lat_ts_nh=lat_ts_nh, lat_ts_sh=lat_ts_sh)
    transform = _affine_from_xy_vectors(x_vec, y_vec)

    # ensure float32
    arr32 = arr.astype("float32", copy=False)

    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
        stereo_tif = os.path.join(tmp_dir, f"{tag}_{hemisphere.lower()}_stereo.tif")
        wgs_tif = os.path.join(tmp_dir, f"{tag}_{hemisphere.lower()}_wgs.tif")

        height, width = arr32.shape
        with rasterio.open(
            stereo_tif,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=arr32.dtype,
            crs=s_srs,
            transform=transform,
        ) as dst:
            dst.write(arr32, 1)

        # nodata string for gdalwarp
        if np.isnan(nodata):
            nodata_str = "nan"
        else:
            nodata_str = str(float(nodata))

        cmd = (
            "gdalwarp -overwrite "
            f"-s_srs '{s_srs}' "
            "-t_srs EPSG:4326 "
            f"-tr {float(grid_resolution_deg)} {float(grid_resolution_deg)} "
            f"-r {resampling} "
            f"-srcnodata {nodata_str} -dstnodata {nodata_str} "
            f"'{stereo_tif}' '{wgs_tif}'"
        )

        subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=True,
        )

        with rasterio.open(wgs_tif) as src:
            data = src.read(1).astype("float32", copy=False)
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


def reproject_vars_polar_to_wgs(
    var_arrays: Dict[str, np.ndarray],
    *,
    hemisphere: str,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    grid_resolution_deg: float,
    lat_ts_nh: float = 70.0,
    lat_ts_sh: float = -71.0,
    resampling: str = "near",
    nodata: float = np.nan,
    tag: str = "polar2wgs",
    tmp_root: Optional[str] = None,
) -> xr.Dataset:
    """
    Reproject multiple polar stereo variables -> WGS84 and merge into xr.Dataset.

    var_arrays: dict[varname -> 2D np.ndarray (y,x)]
    """
    reprojected = []
    for vname, arr in var_arrays.items():
        da = reproject_polar_to_wgs(
            arr,
            hemisphere=hemisphere,
            x_vec=x_vec,
            y_vec=y_vec,
            grid_resolution_deg=grid_resolution_deg,
            lat_ts_nh=lat_ts_nh,
            lat_ts_sh=lat_ts_sh,
            resampling=resampling,
            nodata=nodata,
            tag=f"{tag}_{vname}",
            tmp_root=tmp_root,
        )
        da.name = vname
        reprojected.append(da)

    return xr.merge(reprojected)