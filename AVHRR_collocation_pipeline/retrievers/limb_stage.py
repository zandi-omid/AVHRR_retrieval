from __future__ import annotations

from pathlib import Path
import numpy as np
import xarray as xr

from AVHRR_collocation_pipeline.retrievers.limb_correction.lut_loader import LimbCorrAssets
from AVHRR_collocation_pipeline.retrievers.limb_correction.correction import correct_dataset_vectorized
import AVHRR_collocation_pipeline.utils as utils


class AVHRRLimbCorrectionStage:
    def __init__(
        self,
        *,
        lat_thresh_nh: float,
        lat_thresh_sh: float,
        assets: LimbCorrAssets,
    ):
        self.lat_thresh_nh = float(lat_thresh_nh)
        self.lat_thresh_sh = float(lat_thresh_sh)
        self.assets = assets

    def build_swath_ds_for_limb_correction(
        self,
        *,
        raw_orbit_path: str | Path,
        x_vec: np.ndarray,
        y_vec: np.ndarray,
        land_class_grid: np.ndarray,
        var_name: str = "land_class",
    ) -> xr.Dataset:
        """
        Build an in-memory swath Dataset containing everything required
        for limb correction:
          - latitude, longitude
          - temp_11_0um_nom, temp_12_0um_nom
          - cloud_probability
          - land_class (from AutoSnow grid, nearest-neighbor)
        """
        raw_orbit_path = Path(raw_orbit_path)

        required = [
            "latitude",
            "longitude",
            "temp_11_0um_nom",
            "temp_12_0um_nom",
            "cloud_probability",
        ]

        with xr.open_dataset(raw_orbit_path, decode_timedelta=False) as ds_raw:
            missing = [v for v in required if v not in ds_raw.variables]
            if missing:
                raise KeyError(
                    f"Dataset missing required variables for limb correction: {missing}"
                )

            swath = ds_raw[required].copy()

        # WGS AutoSnow grid
        da_land = xr.DataArray(
            land_class_grid.astype("float32"),
            dims=("y", "x"),
            coords={
                "x": x_vec.astype("float32"),
                "y": y_vec.astype("float32"),
            },
            name=var_name,
        )

        # Nearest-neighbor interpolate AutoSnow → swath
        swath[var_name] = da_land.interp(
            y=swath["latitude"],
            x=swath["longitude"],
            method="nearest",
        ).astype("float32")
        swath[var_name].attrs["coordinates"] = "latitude longitude"

        return swath

    def run(
        self,
        *,
        avh_file: str | Path,
        x_vec: np.ndarray,
        y_vec: np.ndarray,
        land_class_grid: np.ndarray,
    ) -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
        """
        Full limb stage:
          - build swath ds with required vars + land_class (in-memory)
          - run limb correction (in-memory)
          - grid corrected TB11/TB12 back to WGS

        Returns (corrected_swath_ds, tb11_corr_grid, tb12_corr_grid)
        """
        avh_file = Path(avh_file)

        swath_ds = self.build_swath_ds_for_limb_correction(
            raw_orbit_path=avh_file,
            x_vec=x_vec,
            y_vec=y_vec,
            land_class_grid=land_class_grid,
            var_name="land_class",
        )

        swath_ds = correct_dataset_vectorized(
            swath_ds,
            orbit_name=avh_file.name,
            assets=self.assets,
        )

        tb11_corr_grid = utils.grid_swath_var_mean(
            swath_ds, "temp_11_0um_nom_corrected", x_vec, y_vec
        )
        tb12_corr_grid = utils.grid_swath_var_mean(
            swath_ds, "temp_12_0um_nom_corrected", x_vec, y_vec
        )

        return swath_ds, tb11_corr_grid, tb12_corr_grid