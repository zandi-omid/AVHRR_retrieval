from __future__ import annotations

import os
from pathlib import Path

import torch

from AVHRR_collocation_pipeline.readers.AutoSnow_reference import load_AutoSnow_reference
from AVHRR_collocation_pipeline.readers.MERRA2_reference import load_MERRA2_reference

from AVHRR_collocation_pipeline.retrievers.collocate_and_reproj import AVHRRProcessor
from AVHRR_collocation_pipeline.retrievers.retrieve_and_reproj import AVHRRHybridRetriever

import AVHRR_collocation_pipeline.utils as utils

from pytorch_retrieve.architectures import load_model

import xarray as xr
# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
AVHRR_FOLDERS = [
    "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AVHRR/2019",
]

# BASE_OUT = Path("/xdisk/behrangi/omidzandi/retrieved_maps/test_end2end")
BASE_OUT = Path("/xdisk/behrangi/omidzandi/retrieved_maps/test")
BASE_OUT.mkdir(parents=True, exist_ok=True)

GRID_RES = 0.25
LAT_THRESH_NH = 45.0
LAT_THRESH_SH = -45.0
LAT_TS_NH = 70.0
LAT_TS_SH = -71.0

MERRA2_DIR = "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/MERRA2/merra2_archive_19800101_20250831"
AUTOSNOW_DIR = "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AutoSnow/autosnow_in_geotif"
MERRA2_VARS = ["TQV", "T2M"]

# DL inputs (must match what your model expects, in this order)
INPUT_VARS = [
    "cloud_probability",
    "temp_11_0um_nom",
    "temp_12_0um_nom",
    "TQV",
    "T2M",
    "AutoSnow",
]

AVH_VARS = ["cloud_probability", "temp_11_0um_nom", "temp_12_0um_nom"]

# retrieval output format:
#   - "polar": write retrieved fields on polar grid (fastest, no reprojection)
#   - "wgs":   reproject retrieved fields to lat/lon using your polar_to_wgs utilities
# OUT_GRID = "polar"   # change to "wgs" if you want
OUT_GRID = "wgs"   # change to "wgs" if you want

# model checkpoint / loader (placeholder)
CKPT_PATH = (
    "/xdisk/behrangi/omidzandi/DL_Simon_codes/avhrr_retrievals/"
    "checkpoints/AVHRR_efficient_net_v2_pt_1_45_poleward_SH_ERA5_multi_node_keep_all_fp32-v1.ckpt")


def test_end2end_one_orbit() -> None:
    # 1) Pick random AVHRR orbit
    # avh_file = utils.pick_random_nc_file(AVHRR_FOLDERS)
    # orbit_tag = Path(avh_file).stem
    # print(f"\nSelected orbit:\n  {avh_file}\nOrbit tag: {orbit_tag}\n")

    # avh_file = "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AVHRR/2019/clavrx_NSS.GHRR.NN.D19292.S1643.E1838.B7428889.GC.hirs_avhrr_fusion.level2.nc"

    # avh_file = "/xdisk/behrangi/omidzandi/DL_Simon_chips/input_raw_data/AVHRR/2019/clavrx_NSS.GHRR.M1.D19047.S0851.E0948.B3328283.SV.hirs_avhrr_fusion.level2.nc"

    avh_file = "/xdisk/behrangi/omidzandi/DL_Simon_chips/clavrx_NSS.GHRR.NP.D10358.S0247.E0434.B0966970.SV.hirs_avhrr_fusion.level2.nc"

    orbit_tag = Path(avh_file).stem

    print(f"Selected orbit:\n  {avh_file}\nOrbit tag: {orbit_tag}")

    # 2) Load references (only what Stage-1 needs now)
    print("Loading references...")
    merra2_meta = load_MERRA2_reference(MERRA2_DIR)
    autosnow_meta = load_AutoSnow_reference(AUTOSNOW_DIR)

    # 3) Stage-1: collocate + reproject -> ONE polar input NetCDF (NH/SH groups)
    polar_in_nc = BASE_OUT / f"_{orbit_tag}__polar_inputs.nc"

    processor = AVHRRProcessor(
        grid_res=GRID_RES,
        lat_thresh_nh=LAT_THRESH_NH,
        lat_thresh_sh=LAT_THRESH_SH,
        lat_ts_nh=LAT_TS_NH,
        lat_ts_sh=LAT_TS_SH,
        nodata=-9999.0,
        merra2_meta=merra2_meta,
        autosnow_meta=autosnow_meta,
    )

    print("Running Stage-1 (collocate + WGS->polar + write NH/SH groups)...")
    ds_polar = processor.process_orbit(
        avh_file=avh_file,
        avh_vars=AVH_VARS,
        input_vars=INPUT_VARS,
        merra2_vars=MERRA2_VARS,
        out_polar_nc=polar_in_nc,
    )
    print(f"✅ Wrote polar inputs: {polar_in_nc}")

    ds_nh_polar = ds_polar["NH"]
    ds_sh_polar = ds_polar["SH"]

    # 4) Stage-2: retrieval
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CKPT_PATH).to(device).eval()

    retriever = AVHRRHybridRetriever(
        model=model,
        device=device,
        input_vars=INPUT_VARS,
        tile_size=1536,
        overlap=64,
        out_grid_resolution_deg=GRID_RES,
        lat_ts_nh=LAT_TS_NH,
        lat_ts_sh=LAT_TS_SH,
    )

    out_nc = BASE_OUT / f"{orbit_tag}__retrieved_{OUT_GRID}.nc"
    if out_nc.exists():
        out_nc.unlink()

    print(f"Running Stage-2 retrieval (output grid: {OUT_GRID})...")

    # --- Load NH/SH inputs from the polar file ---
    x_nh, tb12_nh, xvec_nh, yvec_nh = retriever.load_group_inputs_from_dataset(ds_nh_polar)
    x_sh, tb12_sh, xvec_sh, yvec_sh = retriever.load_group_inputs_from_dataset(ds_sh_polar)

    # --- GPU inference ---
    preds_nh = retriever.gpu_predict_tiled_multiquantile(x_nh)
    preds_nh = retriever.clean_precip(preds_nh, min_val=0.0, max_val=50.0, drizzle=0.001) # Clean (clip + drizzle removal)
    preds_nh = retriever.mask_preds_with_tb12(preds_nh, tb12_nh) # Mask with TB12 to recover the true swath footprint

    preds_sh = retriever.gpu_predict_tiled_multiquantile(x_sh)
    preds_sh = retriever.clean_precip(preds_sh, min_val=0.0, max_val=50.0, drizzle=0.001)
    preds_sh = retriever.mask_preds_with_tb12(preds_sh, tb12_sh)

    # --- Build datasets to write ---
    if OUT_GRID == "polar":
        # Keep polar coords and write retrieved fields on polar grid
        import xarray as xr
        ds_nh = xr.Dataset(
            {
                "retrieved_precip_mean": (("y", "x"), preds_nh["mean"]),
                "retrieved_precip_q70":  (("y", "x"), preds_nh["q70"]),
                "retrieved_precip_q75":  (("y", "x"), preds_nh["q75"]),
                "retrieved_precip_q80":  (("y", "x"), preds_nh["q80"]),
            },
            coords={"x": xvec_nh, "y": yvec_nh},
        )
        ds_sh = xr.Dataset(
            {
                "retrieved_precip_mean": (("y", "x"), preds_sh["mean"]),
                "retrieved_precip_q70":  (("y", "x"), preds_sh["q70"]),
                "retrieved_precip_q75":  (("y", "x"), preds_sh["q75"]),
                "retrieved_precip_q80":  (("y", "x"), preds_sh["q80"]),
            },
            coords={"x": xvec_sh, "y": yvec_sh},
        )

    elif OUT_GRID == "wgs":
        # Use your central reprojection utils
        from AVHRR_collocation_pipeline.retrievers.reproject import reproject_vars_polar_to_wgs

        var_arrays_nh = {
            "retrieved_precip_mean": preds_nh["mean"],
            "retrieved_precip_q70":  preds_nh["q70"],
            "retrieved_precip_q75":  preds_nh["q75"],
            "retrieved_precip_q80":  preds_nh["q80"],
        }
        var_arrays_sh = {
            "retrieved_precip_mean": preds_sh["mean"],
            "retrieved_precip_q70":  preds_sh["q70"],
            "retrieved_precip_q75":  preds_sh["q75"],
            "retrieved_precip_q80":  preds_sh["q80"],
        }

        ds_nh = reproject_vars_polar_to_wgs(
            var_arrays_nh,
            hemisphere="NH",
            x_vec=xvec_nh,
            y_vec=yvec_nh,
            grid_resolution_deg=GRID_RES,
            lat_ts_nh=LAT_TS_NH,
            lat_ts_sh=LAT_TS_SH,
            nodata=float("nan"),
            tag=f"{orbit_tag}_NH",
        )
        ds_sh = reproject_vars_polar_to_wgs(
            var_arrays_sh,
            hemisphere="SH",
            x_vec=xvec_sh,
            y_vec=yvec_sh,
            grid_resolution_deg=GRID_RES,
            lat_ts_nh=LAT_TS_NH,
            lat_ts_sh=LAT_TS_SH,
            nodata=float("nan"),
            tag=f"{orbit_tag}_SH",
        )

    else:
        raise ValueError("OUT_GRID must be 'polar' or 'wgs'")

    # 5) Write ONE NetCDF with groups NH/SH
    print(f"Writing retrieved NetCDF: {out_nc}")
    retriever.write_orbit_netcdf(
        out_path=out_nc,
        ds_nh=ds_nh,
        ds_sh=ds_sh,
        var_scales={
            "retrieved_precip_mean": 0.005,
            "retrieved_precip_q70":  0.005,
            "retrieved_precip_q75":  0.005,
            "retrieved_precip_q80":  0.005,
        },
        default_scale=0.005,
    )

    print("\n✅ End-to-end test finished.")
    print(f"Polar inputs : {polar_in_nc}")
    print(f"Retrieved nc : {out_nc}")


if __name__ == "__main__":
    test_end2end_one_orbit()