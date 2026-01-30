from __future__ import annotations

from pathlib import Path
import os
import sys
import gc
import time
import socket
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import argparse
import toml

import torch
import xarray as xr

from tqdm import tqdm 

from AVHRR_collocation_pipeline.readers.AutoSnow_reference import load_AutoSnow_reference
from AVHRR_collocation_pipeline.readers.MERRA2_reference import load_MERRA2_reference
from AVHRR_collocation_pipeline.readers.MERRA2_reader import MissingMERRA2File

from AVHRR_collocation_pipeline.retrievers.collocate_and_reproj import AVHRRProcessor
from AVHRR_collocation_pipeline.retrievers.retrieve_and_reproj import AVHRRHybridRetriever
from AVHRR_collocation_pipeline.retrievers.reproject import reproject_vars_polar_to_wgs

from AVHRR_collocation_pipeline.retrievers.back_to_L2 import AVHRRBackToL2

from pytorch_retrieve.architectures import load_model

from AVHRR_collocation_pipeline.retrievers.limb_correction.lut_loader import load_limbcorr_assets

limb_assets = load_limbcorr_assets("/xdisk/behrangi/omidzandi/AVHRR-retrieval/AVHRR_collocation_pipeline/retrievers/limb_correction")

# ------------------------------------------------------------
# Parse config
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/retrieve_config.toml")
args = parser.parse_args()
cfg = toml.load(args.config)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
AVHRR_FOLDERS = cfg["paths"]["avhrr_dirs"]
MERRA2_DIR     = cfg["paths"]["merra2_dir"]
AUTOSNOW_DIR   = cfg["paths"]["autosnow_dir"]

BASE_OUT = Path(cfg["paths"]["out_dir"])
BASE_OUT.mkdir(parents=True, exist_ok=True)

GRID_RES     = float(cfg["grid"]["resolution_deg"])
LAT_THRESH_NH = float(cfg["grid"]["lat_thresh_nh"])
LAT_THRESH_SH = float(cfg["grid"]["lat_thresh_sh"])
LAT_TS_NH     = float(cfg["grid"]["lat_ts_nh"])
LAT_TS_SH     = float(cfg["grid"]["lat_ts_sh"])

CKPT_PATH  = cfg["DL"]["checkpoint"]
TILE_SIZE  = int(cfg["DL"]["tile_size"])
OVERLAP    = int(cfg["DL"]["overlap"])

AVH_VARS      = cfg["input_vars"]["avh_vars"]
MERRA2_VARS   = cfg["input_vars"]["merra2_vars"]
INPUT_VARS    = cfg["input_vars"]["dl_inputs"]

DO_LIMB_CORRECTION = cfg.get("limb_correction", {}).get("do_limb_correction", False)

OUT_GRID = cfg["output"]["grid"].lower()  # "wgs" or "polar"
if OUT_GRID not in ("wgs", "polar"):
    raise ValueError(f"output.grid must be 'wgs' or 'polar', got: {OUT_GRID!r}")

ENABLE_WGS = bool(cfg["output"].get("enable_wgs_output", True))

OUT_CFG = cfg.get("output", {})

WRITE_VARS_NH = OUT_CFG.get("write_vars_nh", None)
WRITE_VARS_SH = OUT_CFG.get("write_vars_sh", None)

RENAME_VARS_NH = OUT_CFG.get("rename_vars_nh", {}) or {}
RENAME_VARS_SH = OUT_CFG.get("rename_vars_sh", {}) or {}

# empty list → write all
if isinstance(WRITE_VARS_NH, list) and len(WRITE_VARS_NH) == 0:
    WRITE_VARS_NH = None
if isinstance(WRITE_VARS_SH, list) and len(WRITE_VARS_SH) == 0:
    WRITE_VARS_SH = None


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def list_avhrr_files(folders: list[str]) -> list[Path]:
    files = []
    for f in folders:
        p = Path(f)
        files.extend(sorted(p.glob("*.nc")))
    return files


def extract_orbit_tag(avh_file: Path) -> str:
    return avh_file.stem

def _filter_and_rename(ds, keep, rename):
    rename = rename or {}

    if keep is not None:
        ds = ds[keep]

    # validate no collisions
    new_names = list(rename.values())
    if len(new_names) != len(set(new_names)):
        raise ValueError(f"Duplicate output variable names after renaming: {new_names}")

    return ds.rename(rename)
# ------------------------------------------------------------
# CPU finalization: reprojection + TB11-mask + NetCDF write
# runs inside CPU thread pool
# ------------------------------------------------------------
def cpu_finalize_orbit(
    *,
    orbit_tag: str,
    raw_orbit_path: Path | str,
    out_nc: Path,
    OUT_GRID: str,
    preds_nh: dict,
    preds_sh: dict,
    xvec_nh,
    yvec_nh,
    xvec_sh,
    yvec_sh,
    tb11_wgs,
    x_vec_global,
    y_vec_global,
    retriever: AVHRRHybridRetriever,
    write_vars_nh: list[str] | None = None,
    write_vars_sh: list[str] | None = None,
    rename_vars_nh: dict[str, str] | None = None,
    rename_vars_sh: dict[str, str] | None = None,
) -> None:
    """
    CPU-only stage:
      - polar -> WGS reprojection (if OUT_GRID == 'wgs')
      - optional TB11 WGS masking
      - write NetCDF with NH / SH groups
    """
    try:
        if out_nc.exists():
            out_nc.unlink()

        # ----------------- build NH/SH datasets -----------------
        if OUT_GRID == "polar":
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
            
            # exporting to nc file
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

        elif OUT_GRID == "wgs":
            # --- polar -> WGS via central utility ---
            # full dicts (unchanged)
            full_nh = {
                "retrieved_precip_mean": preds_nh["mean"],
                "retrieved_precip_q70":  preds_nh["q70"],
                "retrieved_precip_q75":  preds_nh["q75"],
                "retrieved_precip_q80":  preds_nh["q80"],
            }
            full_sh = {
                "retrieved_precip_mean": preds_sh["mean"],
                "retrieved_precip_q70":  preds_sh["q70"],
                "retrieved_precip_q75":  preds_sh["q75"],
                "retrieved_precip_q80":  preds_sh["q80"],
            }

            # choose what to write (defaults = all keys)
            keep_nh = write_vars_nh or list(full_nh.keys())
            keep_sh = write_vars_sh or list(full_sh.keys())

            # filter
            var_arrays_nh = {k: full_nh[k] for k in keep_nh}
            var_arrays_sh = {k: full_sh[k] for k in keep_sh}

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

            # --- mask extra pixels using global TB11 WGS swath ---
            ds_nh = retriever.mask_ds_with_tb11_wgs(ds_nh, tb11_wgs, x_vec_global, y_vec_global)
            ds_sh = retriever.mask_ds_with_tb11_wgs(ds_sh, tb11_wgs, x_vec_global, y_vec_global)

            ds_nh = _filter_and_rename(ds_nh, write_vars_nh, rename_vars_nh)
            ds_sh = _filter_and_rename(ds_sh, write_vars_sh, rename_vars_sh)

        else:
            raise ValueError("OUT_GRID must be 'polar' or 'wgs'")

        if ENABLE_WGS:
            # ----------------- write NetCDF with NH/SH groups (gridded WGS) -----------------
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
        # ----------------- convert gridded WGS retrievals back to L2 swath -------------
        try:
            retrieved_names = list(ds_nh.data_vars) + list(ds_sh.data_vars)
            retrieved_names = sorted(set(retrieved_names))

            l2_writer = AVHRRBackToL2(
                retrieved_var_names=retrieved_names,
            )

            # e.g. same dir as out_nc, different suffix
            l2_out_path = out_nc.with_name(f"{orbit_tag}_L2.nc")

            l2_writer.attach_to_orbit(
                raw_orbit_path=raw_orbit_path,
                ds_nh=ds_nh,
                ds_sh=ds_sh,
                out_path=l2_out_path,
            )

        except Exception as e_l2:
            print(f"[WARN] Could not attach retrievals to L2 grid for {orbit_tag}: {e_l2}", flush=True)

    except Exception as e:
        print(f"❌ [CPU] Error in finalize for {orbit_tag}: {e}", flush=True)
        if out_nc.exists():
            try:
                out_nc.unlink()
            except Exception:
                pass
    finally:
        gc.collect()


# ------------------------------------------------------------
# Per-orbit GPU part (Stage-1 + Stage-2)
# ------------------------------------------------------------
def gpu_stage_for_orbit(
    avh_file: Path,
    processor: AVHRRProcessor,
    retriever: AVHRRHybridRetriever,
    merra2_meta,
    autosnow_meta,
):
    """
    Does:
      - Stage-1: collocate + WGS->polar for this orbit (NH/SH)
      - Stage-2: GPU tiled retrieval for NH/SH

    Returns:
      dict with:
        orbit_tag, preds_nh, preds_sh,
        xvec_nh, yvec_nh, xvec_sh, yvec_sh,
        tb11_wgs, x_vec_global, y_vec_global
    """
    orbit_tag = extract_orbit_tag(avh_file)

    # --- Stage-1: collocate + WGS->polar --- #
    ds_polar, tb11_wgs, x_vec_global, y_vec_global = processor.process_orbit(
        avh_file=avh_file,
        avh_vars=AVH_VARS,
        input_vars=INPUT_VARS,
        merra2_vars=MERRA2_VARS,
        do_limb_correction=DO_LIMB_CORRECTION,
        out_polar_nc="/xdisk/behrangi/omidzandi/retrieved_maps/2010_test/collocated_polar"
    )

    ds_nh_polar = ds_polar["NH"]
    ds_sh_polar = ds_polar["SH"]

    # --- Load NH/SH inputs from polar datasets --- #
    x_nh, tb12_nh, xvec_nh, yvec_nh = retriever.load_group_inputs_from_dataset(ds_nh_polar)
    x_sh, tb12_sh, xvec_sh, yvec_sh = retriever.load_group_inputs_from_dataset(ds_sh_polar)

    # --- GPU inference --- #
    preds_nh = retriever.gpu_predict_tiled_multiquantile(x_nh)
    preds_nh = retriever.clean_precip(preds_nh, min_val=0.0, max_val=50.0, drizzle=0.001)
    preds_nh = retriever.mask_preds_with_tb12(preds_nh, tb12_nh)

    preds_sh = retriever.gpu_predict_tiled_multiquantile(x_sh)
    preds_sh = retriever.clean_precip(preds_sh, min_val=0.0, max_val=50.0, drizzle=0.001)
    preds_sh = retriever.mask_preds_with_tb12(preds_sh, tb12_sh)

    # Free big arrays early (ds_polar will go out of scope when this returns)
    del x_nh, x_sh, ds_nh_polar, ds_sh_polar
    gc.collect()

    return {
        "orbit_tag": orbit_tag,
        "preds_nh": preds_nh,
        "preds_sh": preds_sh,
        "xvec_nh": xvec_nh,
        "yvec_nh": yvec_nh,
        "xvec_sh": xvec_sh,
        "yvec_sh": yvec_sh,
        "tb11_wgs": tb11_wgs,
        "x_vec_global": x_vec_global,
        "y_vec_global": y_vec_global,
    }


# ------------------------------------------------------------
# MAIN: multi-rank + GPU + CPU threadpool
# ------------------------------------------------------------
def main():
    print("=== AVHRR end-to-end retrieval (multi-rank + GPU + CPU threads) ===")

    # ---------- list all AVHRR orbits ---------- #
    all_files = list_avhrr_files(AVHRR_FOLDERS)
    if not all_files:
        print("No AVHRR .nc files found.")
        return

    # ---------- multi-rank splitting (SLURM) ---------- #
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size  = int(os.environ.get("SLURM_NTASKS", 1))
    node_name   = socket.gethostname()

    my_files = all_files[global_rank::world_size]

    print(
        f"[Rank {global_rank}/{world_size}] node={node_name}, "
        f"total_files={len(all_files)}, my_files={len(my_files)}",
        flush=True,
    )

    if not my_files:
        print(f"[Rank {global_rank}] No files assigned, exiting.")
        return

    # ---------- progress logging ---------- #
    progress_file = BASE_OUT / f"progress_rank{global_rank}.log"
    progress_f = open(progress_file, "w")

    # Only rank 0 gets a live tqdm bar (optional)
    if global_rank == 0:
        pbar = tqdm(
            total=len(my_files),
            desc=f"[Rank {global_rank}]",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )
    else:
        pbar = None

    start_time = time.time()

    # ---------- device / model ---------- #
    n_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID", global_rank % max(1, max(n_gpus, 1))))

    if torch.cuda.is_available() and n_gpus > 0:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    torch.set_num_threads(1)

    print(f"[Rank {global_rank}] Using device={device}, GPUs on node={n_gpus}", flush=True)

    model = load_model(CKPT_PATH).to(device).eval()
    retriever = AVHRRHybridRetriever(
        model=model,
        device=device,
        input_vars=INPUT_VARS,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        out_grid_resolution_deg=GRID_RES,
        lat_ts_nh=LAT_TS_NH,
        lat_ts_sh=LAT_TS_SH,
    )

    # ---------- references (shared across orbits on this rank) ---------- #
    print(f"[Rank {global_rank}] Loading references (MERRA2, AutoSnow)...", flush=True)
    merra2_meta = load_MERRA2_reference(MERRA2_DIR)
    autosnow_meta = load_AutoSnow_reference(AUTOSNOW_DIR)

    processor = AVHRRProcessor(
        grid_res=GRID_RES,
        lat_thresh_nh=LAT_THRESH_NH,
        lat_thresh_sh=LAT_THRESH_SH,
        lat_ts_nh=LAT_TS_NH,
        lat_ts_sh=LAT_TS_SH,
        nodata=-9999.0,
        merra2_meta=merra2_meta,
        autosnow_meta=autosnow_meta,
        limb_assets=limb_assets,
    )

    # ---------- CPU thread pool for finalization ---------- #
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    env_reproj = os.environ.get("REPROJ_THREADS", "").strip()
    if env_reproj:
        reproj_threads = int(env_reproj)
    else:
        reproj_threads = max(1, cpus_per_task - 1)

    max_pending = int(os.environ.get("MAX_PENDING_REPROJ", "4"))

    print(
        f"[Rank {global_rank}] Using {reproj_threads} CPU threads + MAX_PENDING_REPROJ={max_pending}",
        flush=True,
    )

    writer_pool = ThreadPoolExecutor(max_workers=reproj_threads)
    futures = []

    start_time = time.time()

    for avh_file in my_files:
        orbit_tag = extract_orbit_tag(avh_file)
        out_nc = BASE_OUT / f"{orbit_tag}_retrieved_{OUT_GRID}.nc"

        print(f"[Rank {global_rank}] >>> Orbit {orbit_tag}", flush=True)

        try:
            # ---------- GPU part (Stage-1 + Stage-2) ---------- #
            gpu_out = gpu_stage_for_orbit(
                avh_file=avh_file,
                processor=processor,
                retriever=retriever,
                merra2_meta=merra2_meta,
                autosnow_meta=autosnow_meta,
            )

            # ---------- CPU part: submit to threadpool ---------- #
            fut = writer_pool.submit(
                cpu_finalize_orbit,
                orbit_tag=gpu_out["orbit_tag"],
                raw_orbit_path=avh_file,
                out_nc=out_nc,
                OUT_GRID=OUT_GRID,
                preds_nh=gpu_out["preds_nh"],
                preds_sh=gpu_out["preds_sh"],
                xvec_nh=gpu_out["xvec_nh"],
                yvec_nh=gpu_out["yvec_nh"],
                xvec_sh=gpu_out["xvec_sh"],
                yvec_sh=gpu_out["yvec_sh"],
                tb11_wgs=gpu_out["tb11_wgs"],
                x_vec_global=gpu_out["x_vec_global"],
                y_vec_global=gpu_out["y_vec_global"],
                retriever=retriever,
                write_vars_nh=WRITE_VARS_NH,
                write_vars_sh=WRITE_VARS_SH,
                rename_vars_nh=RENAME_VARS_NH,
                rename_vars_sh=RENAME_VARS_SH,
            )
            futures.append(fut)

            # Drop references from main thread (thread has its own copies)
            del gpu_out
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # throttle number of pending CPU jobs
            if len(futures) >= max_pending:
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                futures = list(not_done)

        except MissingMERRA2File as e:
            print(f"[Rank {global_rank}] [SKIP] {orbit_tag}: {e}", file=sys.stderr, flush=True)
            if out_nc.exists():
                try:
                    out_nc.unlink()
                except Exception:
                    pass

            continue

        except Exception as e:
            print(f"[Rank {global_rank}] ❌ Error on orbit {orbit_tag}: {e}", flush=True)
            if out_nc.exists():
                try:
                    out_nc.unlink()
                except Exception:
                    pass

        # ---------- progress logging ---------- #
        completed = (pbar.n + 1) if pbar is not None else None
        total = len(my_files)

        # Update tqdm (if enabled for this rank)
        if pbar is not None:
            pbar.update(1)

        # Fallback for ranks without tqdm
        if completed is None:
            completed = my_files.index(avh_file) + 1

        progress = completed / total
        elapsed = time.time() - start_time
        its_per_sec = completed / elapsed if elapsed > 0 else 0.0

        # --- ETA computation (per rank) ---
        if its_per_sec > 0.0:
            remaining = total - completed
            eta_sec = remaining / its_per_sec
            # simple formatting: hours + minutes
            eta_hours = int(eta_sec // 3600)
            eta_minutes = int((eta_sec % 3600) // 60)
            if eta_hours > 0:
                eta_str = f"{eta_hours}h {eta_minutes:02d}m"
            else:
                eta_str = f"{eta_minutes}m"
        else:
            eta_str = "--"

        # --- textual progress bar for log file ---
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "#" * filled + "-" * (bar_width - filled)

        progress_f.write(
            f"{completed:5d}/{total:5d} [{bar}] "
            f"{progress*100:5.1f}% | {its_per_sec:5.2f} it/s | "
            f"ETA {eta_str} | {orbit_tag}\n"
        )
        progress_f.flush()

    # ---------- wait for CPU jobs ---------- #
    if futures:
        wait(futures)
    writer_pool.shutdown(wait=True)

    if 'pbar' in locals() and pbar is not None:
        pbar.close()
    progress_f.close()

    # cleanup
    del model, retriever, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dt = time.time() - start_time

    print(f"[Rank {global_rank}] Done all orbits in {dt/3600:.2f} hours.", flush=True)


if __name__ == "__main__":
    main()