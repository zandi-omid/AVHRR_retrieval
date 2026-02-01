from __future__ import annotations

from pathlib import Path
from typing import Sequence, Dict, Optional

import xarray as xr


class AVHRRBackToL2:
    """
    Attach gridded retrievals (NH/SH WGS grids) back onto the original
    AVHRR L2 orbit swath grid.

    Workflow per orbit:
      1) Read original L2 orbit (L2 swath file).
      2) Merge NH + SH retrieval datasets into a single global WGS grid.
      3) Interpolate retrievals at each pixel's (lat, lon) from the raw orbit.
      4) Build a new L2-like dataset with:
           - latitude, longitude
           - scan_line_time
           - temp_11_0um_nom
           - temp_12_0um_nom
           - retrieved_precip_mean, q70, q75, q80 (or custom list)
      5) Save to NetCDF.
    """

    def __init__(
        self,
        retrieved_var_names: Sequence[str] = (
            "retrieved_precip_mean",
            "retrieved_precip_q70",
            "retrieved_precip_q75",
            "retrieved_precip_q80",
        ),
        line_dim: str = "scan_lines_along_track_direction",
        pix_dim: str = "pixel_elements_along_scan_direction",

    ) -> None:
        """
        Parameters
        ----------
        retrieved_var_names
            Names of retrieval variables that exist in ds_nh/ds_sh
            and should be interpolated back to the swath grid.
        line_dim
            Name of the along-track dimension in the raw L2 file.
        pix_dim
            Name of the cross-track dimension in the raw L2 file.
        """
        self.retrieved_var_names = list(retrieved_var_names)
        self.line_dim = line_dim
        self.pix_dim = pix_dim

    # ------------------------------------------------------------------
    # Internal helper: merge hemispheric grids
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_hemispheric_grids(
        ds_nh: xr.Dataset,
        ds_sh: xr.Dataset,
        var_names: Sequence[str],
    ) -> Dict[str, xr.DataArray]:
        """
        Merge NH and SH WGS grids into a single global WGS grid
        for each variable. Assumes:

          - dims: (y, x)
          - coords: y ~ latitude, x ~ longitude
          - NH and SH do not overlap in latitude.

        Returns
        -------
        dict
            {var_name: global_DataArray}
        """
        global_vars: Dict[str, xr.DataArray] = {}

        for v in var_names:
            if v not in ds_nh and v not in ds_sh:
                print(f"[WARN] Retrieval variable '{v}' not in ds_nh or ds_sh — skipping.")
                continue

            da_nh: Optional[xr.DataArray] = ds_nh[v] if v in ds_nh else None
            da_sh: Optional[xr.DataArray] = ds_sh[v] if v in ds_sh else None

            if da_nh is not None and da_sh is not None:
                # combine_first prefers non-NaN from da_nh, fills with da_sh
                global_da = da_nh.combine_first(da_sh)
            elif da_nh is not None:
                global_da = da_nh
            else:
                global_da = da_sh  # type: ignore[assignment]

            global_vars[v] = global_da

        return global_vars

    def attach_to_orbit_ds(
        self,
        raw_orbit_path: Path | str,
        ds_nh: xr.Dataset,
        ds_sh: xr.Dataset,
        copy_vars_from_raw: list[str] | None = None,
    ) -> xr.Dataset:
        raw_orbit_path = Path(raw_orbit_path)
        ds_raw = xr.open_dataset(raw_orbit_path, decode_timedelta=True)

        copy_vars_from_raw = copy_vars_from_raw or []

        line_dim = self.line_dim
        pix_dim = self.pix_dim

        lat_raw = ds_raw["latitude"]
        lon_raw = ds_raw["longitude"]

        out = xr.Dataset()
        out = out.assign_coords(
            {
                line_dim: lat_raw.coords[line_dim] if line_dim in lat_raw.coords else range(ds_raw.sizes[line_dim]),
                pix_dim:  lat_raw.coords[pix_dim]  if pix_dim  in lat_raw.coords else range(ds_raw.sizes[pix_dim]),
                "latitude":  lat_raw,
                "longitude": lon_raw,
            }
        )

        # Always keep time/TBs if present (your current behavior)

        out["scan_line_time"] = ds_raw["scan_line_time"]
        out["temp_11_0um_nom"] = ds_raw["temp_11_0um_nom"]
        out["temp_12_0um_nom"] = ds_raw["temp_12_0um_nom"]

        # copy additional raw vars needed by limb correction (e.g., cloud_probability)
        for v in copy_vars_from_raw:
            if v in ds_raw:
                out[v] = ds_raw[v]
            else:
                print(f"[WARN] raw orbit missing '{v}' — limb correction may fail.", flush=True)

        # Merge gridded land_class (or retrieval vars) onto swath grid
        global_grids = self._merge_hemispheric_grids(ds_nh, ds_sh, self.retrieved_var_names)

        for v in self.retrieved_var_names:
            if v not in global_grids:
                continue
            da_global = global_grids[v]
            da_on_swath = da_global.interp(y=lat_raw, x=lon_raw, method="nearest")

            rename_map = {}
            if len(da_on_swath.dims) == 2:
                d0, d1 = da_on_swath.dims
                if d0 != line_dim:
                    rename_map[d0] = line_dim
                if d1 != pix_dim:
                    rename_map[d1] = pix_dim
            if rename_map:
                da_on_swath = da_on_swath.rename(rename_map)

            da_on_swath.attrs["coordinates"] = "latitude longitude"
            out[v] = da_on_swath

        ds_raw.close()
        return out

    def attach_to_orbit(
        self,
        raw_orbit_path: Path | str,
        ds_nh: xr.Dataset,
        ds_sh: xr.Dataset,
        out_path: Path | str,
    ) -> Path:
        out_path = Path(out_path)
        out_ds = self.attach_to_orbit_ds(raw_orbit_path, ds_nh, ds_sh)

        encoding = {}
        for v in self.retrieved_var_names:
            if v in out_ds:
                encoding[v] = {
                    "dtype": "float32",
                    "_FillValue": float("nan"),
                    "zlib": True,
                    "complevel": 4,
                }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_ds.to_netcdf(out_path, format="NETCDF4", encoding=encoding)
        return out_path