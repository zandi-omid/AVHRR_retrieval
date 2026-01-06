from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
import torch

from AVHRR_collocation_pipeline.retrievers.netcdf_encoding import build_uint16_encoding
from AVHRR_collocation_pipeline.retrievers.reproject.polar_to_wgs import (
    reproject_vars_polar_to_wgs,
)


class AVHRRHybridRetriever:
    """
    Stage-2 Hybrid Retriever:
      polar-stereo inputs (NH/SH groups) -> GPU tiled inference -> (optional) polar->WGS reproject -> one NetCDF

    - No IMERG / ERA5
    - Writes one output NetCDF with groups: NH and SH
    - All reprojection functions live under retrievers/reproject/
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        input_vars: list[str],
        tile_size: int = 1536,
        overlap: int = 64,
        out_grid_resolution_deg: float = 0.25,
        lat_ts_nh: float = 70.0,
        lat_ts_sh: float = -71.0,
    ):
        self.model = model.eval().to(device)
        self.device = device
        self.input_vars = input_vars

        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.out_grid_resolution_deg = float(out_grid_resolution_deg)

        self.lat_ts_nh = float(lat_ts_nh)
        self.lat_ts_sh = float(lat_ts_sh)

        if "temp_12_0um_nom" not in self.input_vars:
            raise ValueError("'temp_12_0um_nom' must be included in input_vars (used as swath mask).")

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def load_group_inputs(
        self,
        file_path: Path,
        group: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read one group (NH/SH) from a polar-stereo NetCDF.

        Returns:
          x_stack: [C,H,W] float32 (DL inputs)
          tb12   : [H,W]   float32 (mask)
          x_vec  : [W]     float64/float32
          y_vec  : [H]     float64/float32
        """
        with xr.open_dataset(file_path, group=group, decode_cf=True) as ds:
            for v in self.input_vars:
                if v not in ds:
                    raise KeyError(f"[{group}] missing required input var: {v}")

            x_stack = np.stack([ds[v].values.astype(np.float32) for v in self.input_vars], axis=0)
            tb12 = ds["temp_12_0um_nom"].values.astype(np.float32)
            x_vec = ds["x"].values
            y_vec = ds["y"].values

        return x_stack, tb12, x_vec, y_vec

    # ------------------------------------------------------------------
    # GPU tiled inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def gpu_predict_tiled_multiquantile(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        x: [C,H,W] float32 (NaNs outside swath)
        returns dict(mean,q70,q75,q80) each [H,W] float32
        """
        tile = self.tile_size
        overlap = self.overlap
        step = tile - overlap

        _, H, W = x.shape
        tb12 = x[self.input_vars.index("temp_12_0um_nom")]
        valid = np.isfinite(tb12)

        if not valid.any():
            return {k: np.full((H, W), np.nan, dtype=np.float32) for k in ["mean", "q70", "q75", "q80"]}

        rows = np.where(valid.any(axis=1))[0]
        cols = np.where(valid.any(axis=0))[0]
        ymin, ymax = rows.min(), rows.max()
        xmin, xmax = cols.min(), cols.max()

        x_crop = x[:, ymin:ymax + 1, xmin:xmax + 1]
        tb12_crop = tb12[ymin:ymax + 1, xmin:xmax + 1]
        Hc, Wc = x_crop.shape[1:]

        pad_h = (step - (Hc % step)) % step
        pad_w = (step - (Wc % step)) % step
        if pad_h or pad_w:
            x_crop = np.pad(x_crop, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
            tb12_crop = np.pad(tb12_crop, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)

        Hcp, Wcp = x_crop.shape[1:]
        results_pad = {k: np.zeros((Hcp, Wcp), dtype=np.float32) for k in ["mean", "q70", "q75", "q80"]}
        weight = np.zeros((Hcp, Wcp), dtype=np.float32)

        for i in range(0, Hcp, step):
            for j in range(0, Wcp, step):
                tile_arr = x_crop[:, i:i + tile, j:j + tile]
                th, tw = tile_arr.shape[1:]
                if th == 0 or tw == 0:
                    continue

                chip_tb = tb12_crop[i:i + th, j:j + tw]
                if np.isfinite(chip_tb).mean() < 0.05:
                    continue

                th -= th % 32
                tw -= tw % 32
                if th == 0 or tw == 0:
                    continue

                xt = torch.from_numpy(tile_arr[:, :th, :tw]).unsqueeze(0).to(self.device)

                if self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model(xt)
                else:
                    out = self.model(xt)

                mean_tile = out.expected_value().squeeze().detach().float().cpu().numpy()
                qs = out.quantiles([0.7, 0.75, 0.8]).detach().cpu().numpy()

                q70 = qs[:, 0, ...].squeeze()
                q75 = qs[:, 1, ...].squeeze()
                q80 = qs[:, 2, ...].squeeze()

                results_pad["mean"][i:i + th, j:j + tw] += mean_tile
                results_pad["q70"][i:i + th, j:j + tw] += q70
                results_pad["q75"][i:i + th, j:j + tw] += q75
                results_pad["q80"][i:i + th, j:j + tw] += q80
                weight[i:i + th, j:j + tw] += 1.0

                del xt, out, mean_tile, qs, q70, q75, q80
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        weight_safe = np.where(weight == 0, np.nan, weight)
        for k in results_pad:
            results_pad[k] /= weight_safe

        results_crop = {k: v[:Hc, :Wc] for k, v in results_pad.items()}
        results_full = {k: np.full((H, W), np.nan, dtype=np.float32) for k in ["mean", "q70", "q75", "q80"]}
        for k in results_full:
            results_full[k][ymin:ymax + 1, xmin:xmax + 1] = results_crop[k]

        return results_full

    # ------------------------------------------------------------------
    # Build xr.Dataset for one group (polar grid)
    # ------------------------------------------------------------------
    @staticmethod
    def _mask_preds(preds: Dict[str, np.ndarray], tb12: np.ndarray) -> Dict[str, np.ndarray]:
        mask = ~np.isfinite(tb12)
        return {
            "retrieved_precip_mean": np.where(mask, np.nan, preds["mean"]).astype(np.float32),
            "retrieved_precip_q70":  np.where(mask, np.nan, preds["q70"]).astype(np.float32),
            "retrieved_precip_q75":  np.where(mask, np.nan, preds["q75"]).astype(np.float32),
            "retrieved_precip_q80":  np.where(mask, np.nan, preds["q80"]).astype(np.float32),
        }

    def build_group_dataset_polar(
        self,
        *,
        preds: Dict[str, np.ndarray],
        x_vec: np.ndarray,
        y_vec: np.ndarray,
        tb12: np.ndarray,
    ) -> xr.Dataset:
        var_arrays = self._mask_preds(preds, tb12)

        ds = xr.Dataset(
            {}
        ).assign_coords(
            x=np.asarray(x_vec),
            y=np.asarray(y_vec),
        )

        for vname, arr in var_arrays.items():
            ds[vname] = (("y", "x"), arr)

        return ds

    # ------------------------------------------------------------------
    # One-orbit writer: one NetCDF with NH/SH groups
    # ------------------------------------------------------------------
    def write_orbit_netcdf(
        self,
        *,
        out_path: Path,
        ds_nh: xr.Dataset,
        ds_sh: xr.Dataset,
        var_scales: Optional[dict[str, float]] = None,
        default_scale: float = 0.005,
    ) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        enc_nh = build_uint16_encoding(ds_nh, default_scale=default_scale, var_scales=var_scales or {})
        ds_nh.to_netcdf(out_path, mode="w", group="NH", format="NETCDF4", encoding=enc_nh)

        enc_sh = build_uint16_encoding(ds_sh, default_scale=default_scale, var_scales=var_scales or {})
        ds_sh.to_netcdf(out_path, mode="a", group="SH", format="NETCDF4", encoding=enc_sh)

    # ------------------------------------------------------------------
    # Full orbit run (input polar file -> output netcdf)
    # ------------------------------------------------------------------
    def run_orbit(
        self,
        *,
        polar_input_nc: Path,
        out_nc: Path,
        output_in_wgs: bool = True,
        resampling: str = "near",
        nodata: float = np.nan,
        encoding_scales: Optional[dict[str, float]] = None,
        default_scale: float = 0.005,
        tag: Optional[str] = None,
    ) -> Path:
        """
        Main entrypoint:
          - loads NH/SH inputs from one polar file
          - runs GPU inference for NH/SH
          - optionally reprojects outputs to WGS
          - writes one output NetCDF with NH/SH groups

        output_in_wgs=True:
          output grids are EPSG:4326 with x/y in degrees

        output_in_wgs=False:
          output stays polar stereographic with x/y in meters
        """
        polar_input_nc = Path(polar_input_nc)
        out_nc = Path(out_nc)
        tag = tag or polar_input_nc.stem

        # --- NH ---
        x_nh, tb12_nh, xvec_nh, yvec_nh = self.load_group_inputs(polar_input_nc, "NH")
        preds_nh = self.gpu_predict_tiled_multiquantile(x_nh)
        ds_nh_polar = self.build_group_dataset_polar(preds=preds_nh, x_vec=xvec_nh, y_vec=yvec_nh, tb12=tb12_nh)

        # --- SH ---
        x_sh, tb12_sh, xvec_sh, yvec_sh = self.load_group_inputs(polar_input_nc, "SH")
        preds_sh = self.gpu_predict_tiled_multiquantile(x_sh)
        ds_sh_polar = self.build_group_dataset_polar(preds=preds_sh, x_vec=xvec_sh, y_vec=yvec_sh, tb12=tb12_sh)

        if output_in_wgs:
            ds_nh = reproject_vars_polar_to_wgs(
                {v: ds_nh_polar[v].values for v in ds_nh_polar.data_vars},
                hemisphere="NH",
                x_vec=xvec_nh,
                y_vec=yvec_nh,
                grid_resolution_deg=self.out_grid_resolution_deg,
                lat_ts_nh=self.lat_ts_nh,
                lat_ts_sh=self.lat_ts_sh,
                resampling=resampling,
                nodata=nodata,
                tag=f"{tag}_NH",
            )
            ds_sh = reproject_vars_polar_to_wgs(
                {v: ds_sh_polar[v].values for v in ds_sh_polar.data_vars},
                hemisphere="SH",
                x_vec=xvec_sh,
                y_vec=yvec_sh,
                grid_resolution_deg=self.out_grid_resolution_deg,
                lat_ts_nh=self.lat_ts_nh,
                lat_ts_sh=self.lat_ts_sh,
                resampling=resampling,
                nodata=nodata,
                tag=f"{tag}_SH",
            )
        else:
            ds_nh = ds_nh_polar
            ds_sh = ds_sh_polar

        self.write_orbit_netcdf(
            out_path=out_nc,
            ds_nh=ds_nh,
            ds_sh=ds_sh,
            var_scales=encoding_scales,
            default_scale=default_scale,
        )

        # free memory
        del x_nh, x_sh, preds_nh, preds_sh, ds_nh_polar, ds_sh_polar, ds_nh, ds_sh
        gc.collect()

        return out_nc