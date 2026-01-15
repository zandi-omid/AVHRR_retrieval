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

    # ------------------------------------------------------------------
    # Public method: attach retrieval to L2 orbit
    # ------------------------------------------------------------------
    def attach_to_orbit(
        self,
        raw_orbit_path: Path | str,
        ds_nh: xr.Dataset,
        ds_sh: xr.Dataset,
        out_path: Path | str,
    ) -> Path:
        """
        Read the original L2 orbit, sample retrievals from ds_nh/ds_sh
        onto the swath grid, and write a new L2-like file containing:

          - latitude, longitude
          - scan_line_time
          - temp_11_0um_nom
          - temp_12_0um_nom
          - self.retrieved_var_names

        All 2D vars end up with shape:
          (scan_lines_along_track_direction, pixel_elements_along_scan_direction)

        Many values will be NaN where the retrieval has no coverage
        (e.g., outside ±45° poleward domain).

        Parameters
        ----------
        raw_orbit_path
            Path to the original L2 orbit NetCDF file.
        ds_nh
            Retrieval dataset for the Northern Hemisphere (WGS, dims (y, x)).
        ds_sh
            Retrieval dataset for the Southern Hemisphere (WGS, dims (y, x)).
        out_path
            Path to output L2-like NetCDF file.

        Returns
        -------
        Path
            The path of the written file.
        """
        raw_orbit_path = Path(raw_orbit_path)
        out_path = Path(out_path)

        # ----------------- 1. open original orbit -----------------
        ds_raw = xr.open_dataset(raw_orbit_path)

        line_dim = self.line_dim
        pix_dim = self.pix_dim

        lat_raw = ds_raw["latitude"]   # (lines, pixels)
        lon_raw = ds_raw["longitude"]  # (lines, pixels)

        # ----------------- 2. merge NH+SH into global WGS grid -----------------
        global_grids = self._merge_hemispheric_grids(ds_nh, ds_sh, self.retrieved_var_names)

        # ----------------- 3. build new L2-style dataset -----------------
        out = xr.Dataset()

        # Keep latitude/longitude as coordinates on swath grid
        out = out.assign_coords(
            {
                line_dim: lat_raw.coords[line_dim] if line_dim in lat_raw.coords else range(ds_raw.dims[line_dim]),
                pix_dim: lat_raw.coords[pix_dim] if pix_dim in lat_raw.coords else range(ds_raw.dims[pix_dim]),
                "latitude": lat_raw,
                "longitude": lon_raw,
            }
        )

        # Keep the time + TBs (if present)
        if "scan_line_time" in ds_raw:
            out["scan_line_time"] = ds_raw["scan_line_time"]

        if "temp_11_0um_nom" in ds_raw:
            out["temp_11_0um_nom"] = ds_raw["temp_11_0um_nom"]

        if "temp_12_0um_nom" in ds_raw:
            out["temp_12_0um_nom"] = ds_raw["temp_12_0um_nom"]

        # ----------------- 4. interpolate retrievals to swath coords -----------------
        for v in self.retrieved_var_names:
            if v not in global_grids:
                print(f"[WARN] Retrieval variable '{v}' not found in merged global grids — skipping.")
                continue

            da_global = global_grids[v]

            # da_global has dims (y, x) with coordinates y,x in degrees
            # lat_raw, lon_raw have dims (line_dim, pix_dim) in degrees
            # We can interpolate by matching y->latitude, x->longitude
            da_on_swath = da_global.interp(
                y=lat_raw,
                x=lon_raw,
                method="nearest",
            )

            # The result dims will already be (line_dim, pix_dim) because it
            # takes dims from lat_raw/lon_raw, but we sanitize just in case.
            rename_map = {}
            if len(da_on_swath.dims) == 2:
                d0, d1 = da_on_swath.dims
                if d0 != line_dim:
                    rename_map[d0] = line_dim
                if d1 != pix_dim:
                    rename_map[d1] = pix_dim

            if rename_map:
                da_on_swath = da_on_swath.rename(rename_map)

            out[v] = da_on_swath

        # Optional: carry some global attrs from original and add a note
        out.attrs.update(ds_raw.attrs)
        out.attrs.update(
            {
                "source_orbit_file": str(raw_orbit_path),
                "comment": "AVHRR orbit with attached retrieved precipitation fields on swath grid.",
            }
        )

        # ----------------- 5. write to NetCDF -----------------
        out_path.parent.mkdir(parents=True, exist_ok=True)

        encoding = {}

        # 1) retrieved precip fields → uint16 @ precip_scale
        for v in self.retrieved_var_names:
            if v in out.data_vars:
                encoding[v] = {
                    "dtype": "uint16",
                    "scale_factor": self.precip_scale,
                    "_FillValue": 65535,
                    "zlib": True,
                    "complevel": 4,
                }

        # 2) TB variables → uint16 @ tb_scale
        for v in ("temp_11_0um_nom", "temp_12_0um_nom"):
            if v in out.data_vars:
                encoding[v] = {
                    "dtype": "uint16",
                    "scale_factor": self.tb_scale,
                    "_FillValue": 65535,
                    "zlib": True,
                    "complevel": 4,
                }

        # 3) scan_line_time → stored as-is (NO SCALE)
        #    Do not add it to encoding, xarray leaves dtype unchanged

        out.to_netcdf(out_path, format="NETCDF4", encoding=encoding)

        ds_raw.close()
        return out_path