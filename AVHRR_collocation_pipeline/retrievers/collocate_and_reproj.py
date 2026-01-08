from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

# --- readers ---
from AVHRR_collocation_pipeline.readers.AVHRR_reader import read_AVHRR_orbit_to_df
from AVHRR_collocation_pipeline.readers.MERRA2_reader import collocate_MERRA2
from AVHRR_collocation_pipeline.readers.AutoSnow_reader import collocate_AutoSnow

# --- reprojection (new location) ---
from AVHRR_collocation_pipeline.retrievers.reproject import reproject_vars_wgs_to_polar

# --- netcdf encoding (for 1 file with NH/SH groups) ---
from AVHRR_collocation_pipeline.retrievers.netcdf_encoding import build_uint16_encoding

# --- utils ---
import AVHRR_collocation_pipeline.utils as utils


class AVHRRProcessor:
    """
    Stage-1 processor for AVHRR (clean version):

      raw AVHRR orbit (WGS)
        -> collocate MERRA2 + AutoSnow (DL features)
        -> build WGS grids
        -> reproject WGS grids to polar stereographic (NH/SH)
        -> write ONE polar NetCDF with groups: NH and SH

    This prepares the exact input file needed by Stage-2 (AVHRRHybridRetriever).
    """

    def __init__(
        self,
        *,
        grid_res: float,
        lat_thresh_nh: float = 55.0,
        lat_thresh_sh: float = -55.0,
        lat_ts_nh: float = 70.0,
        lat_ts_sh: float = -71.0,
        nodata: float = -9999.0,
        merra2_meta=None,
        autosnow_meta=None,
    ):
        self.grid_res = float(grid_res)
        self.lat_thresh_nh = float(lat_thresh_nh)
        self.lat_thresh_sh = float(lat_thresh_sh)
        self.lat_ts_nh = float(lat_ts_nh)
        self.lat_ts_sh = float(lat_ts_sh)
        self.nodata = float(nodata)

        self.merra2_meta = merra2_meta
        self.autosnow_meta = autosnow_meta

    # --------------------------------------------------------
    # 1) Read orbit to DataFrame
    # --------------------------------------------------------
    def load_orbit_df(self, avh_file: str, avh_vars: List[str]):
        """
        Wrapper around read_AVHRR_orbit_to_df + add_time_columns.
        Also builds the regular WGS grid vectors.
        """
        x_vec, y_vec, x, y = utils.build_test_grid(self.grid_res)

        # store grid for later use in df2grid
        self._x_grid = x
        self._y_grid = y

        df = read_AVHRR_orbit_to_df(
            avh_file,
            x_vec,
            y_vec,
            x,
            y,
            avh_vars=avh_vars,
            lat_thresh_N_hemisphere=self.lat_thresh_nh,
            lat_thresh_S_hemisphere=self.lat_thresh_sh,
        )

        if df is None or df.empty:
            print(f"[WARN] AVHRR reader returned empty for {avh_file}")
            return None, None, None

        df = utils.add_time_columns(df)
        return df, x_vec, y_vec

    # --------------------------------------------------------
    # 2) Collocate reference datasets (ONLY what DL needs)
    # --------------------------------------------------------
    def collocate_dl_features(
        self,
        df,
        *,
        merra2_vars: List[str],
    ):
        """
        Collocate ONLY the features required by the DL model:
          - MERRA2 (e.g., TQV, T2M)
          - AutoSnow

        IMERG/ERA5 are intentionally removed from Stage-1.
        """
        if self.merra2_meta is not None and merra2_vars:
            df = collocate_MERRA2(df, self.merra2_meta, MERRA2_vars=merra2_vars)

        if self.autosnow_meta is not None:
            df = collocate_AutoSnow(df, self.autosnow_meta, date_col="scan_date", out_col="AutoSnow")

        return df

    # --------------------------------------------------------
    # 3) Build 2D WGS grids for selected variables
    # --------------------------------------------------------
    def build_var_grids(self, df, x_vec, y_vec, varnames: List[str]) -> Dict[str, np.ndarray]:
        if not hasattr(self, "_x_grid") or not hasattr(self, "_y_grid"):
            raise RuntimeError("x/y grid not set on AVHRRProcessor. Did you call load_orbit_df first?")

        var_grids: Dict[str, np.ndarray] = {}
        for v in varnames:
            if v not in df.columns:
                print(f"[WARN] '{v}' not found in df — skipping grid")
                continue

            grid = utils.df2grid(
                df,
                v,
                x_vec=x_vec,
                y_vec=y_vec,
                x=self._x_grid,
                y=self._y_grid,
            ).astype("float32")

            var_grids[v] = grid

        return var_grids

    # --------------------------------------------------------
    # 4) Reproject WGS grids → polar stereo (NH + SH)
    # --------------------------------------------------------
    def wgs_to_polar(self, var_grids: Dict[str, np.ndarray], orbit_tag: str, x_vec, y_vec):
        return reproject_vars_wgs_to_polar(
            var_grids,
            orbit_tag=orbit_tag,
            x_vec=x_vec,
            y_vec=y_vec,
            grid_resolution=self.grid_res,
            lat_thresh_nh=self.lat_thresh_nh,
            lat_thresh_sh=self.lat_thresh_sh,
            lat_ts_nh=self.lat_ts_nh,
            lat_ts_sh=self.lat_ts_sh,
            nodata=self.nodata
        )

    # --------------------------------------------------------
    # 5) Write ONE NetCDF with NH/SH groups
    # --------------------------------------------------------
    def write_polar_groups_netcdf(
        self,
        *,
        polar: Dict,
        out_nc: Path,
        var_scales: Optional[Dict[str, float]] = None,
        default_scale: float = 0.005,
    ) -> Path:
        """
        polar is expected to include:
          polar["NH"], polar["SH"] as dict of {var: (("y","x"), arr)}
        and (because return_coords=True):
          polar["coords"]["NH"]["x"], ["y"], same for SH
        """
        out_nc = Path(out_nc)
        out_nc.parent.mkdir(parents=True, exist_ok=True)

        if "coords" not in polar:
            raise ValueError("polar dict is missing 'coords'. Call wgs_to_polar(..., return_coords=True).")

        def _to_ds(hemi: str) -> xr.Dataset:
            ds = xr.Dataset(polar[hemi])
            ds = ds.assign_coords(
                x=np.asarray(polar["coords"][hemi]["x"]),
                y=np.asarray(polar["coords"][hemi]["y"]),
            )
            return ds

        ds_nh = _to_ds("NH")
        ds_sh = _to_ds("SH")

        enc_nh = build_uint16_encoding(ds_nh, default_scale=default_scale, var_scales=var_scales or {})
        ds_nh.to_netcdf(out_nc, mode="w", group="NH", format="NETCDF4", encoding=enc_nh)

        enc_sh = build_uint16_encoding(ds_sh, default_scale=default_scale, var_scales=var_scales or {})
        ds_sh.to_netcdf(out_nc, mode="a", group="SH", format="NETCDF4", encoding=enc_sh)

        return out_nc

    def _polar_to_dataset(self, polar: Dict, hemi: str) -> xr.Dataset:
        """
        Convert the 'polar' dict returned by wgs_to_polar into an
        xarray.Dataset for a given hemisphere ("NH" or "SH").
        """
        ds = xr.Dataset(polar[hemi])
        ds = ds.assign_coords(
            x=np.asarray(polar["coords"][hemi]["x"]),
            y=np.asarray(polar["coords"][hemi]["y"]),
        )
        return ds

    # --------------------------------------------------------
    # 6) One-step pipeline for a single orbit
    # --------------------------------------------------------
    def process_orbit(
        self,
        avh_file: str,
        *,
        avh_vars: List[str],
        input_vars: List[str],
        merra2_vars: List[str],
        out_polar_nc: Optional[str | Path] = None,
        encoding_scales: Optional[Dict[str, float]] = None,
        default_scale: float = 0.005,
    ):
        """
        Full Stage-1 pipeline for a single AVHRR orbit.

        Returns
        -------
        polar : dict
            Same structure as reproject_vars_wgs_to_polar(..., return_coords=True)

        If out_polar_nc is provided, also writes ONE netcdf file with groups NH/SH.
        """
        # 1) Read AVHRR orbit
        df, x_vec, y_vec = self.load_orbit_df(avh_file, avh_vars)
        if df is None:
            return None

        # 2) Collocate DL features (no IMERG/ERA5)
        df = self.collocate_dl_features(df, merra2_vars=merra2_vars)

        # df.to_pickle('/xdisk/behrangi/omidzandi/retrieved_maps/test_dfs/df_new.pkl')

        orbit_tag = Path(avh_file).stem

        # 3) Grid required DL inputs
        var_grids = self.build_var_grids(df, x_vec, y_vec, input_vars)

        np.savez("/xdisk/behrangi/omidzandi/retrieved_maps/test/2010_new_var_grids_debug.npz", **var_grids)

        # 4) Reproject to polar (still returns the dict)
        polar = self.wgs_to_polar(var_grids, orbit_tag, x_vec, y_vec)

        # 5) Optionally write ONE polar NetCDF with NH/SH groups (no change)
        if out_polar_nc is not None:
            self.write_polar_groups_netcdf(
                polar=polar,
                out_nc=Path(out_polar_nc),
                var_scales=encoding_scales,
                default_scale=default_scale,
            )

        # 6) Build in-memory datasets for NH / SH and return those
        ds_nh = self._polar_to_dataset(polar, "NH")
        ds_sh = self._polar_to_dataset(polar, "SH")

        # You can return a dict or a tuple; I'll use a dict:
        return {"NH": ds_nh, "SH": ds_sh}
