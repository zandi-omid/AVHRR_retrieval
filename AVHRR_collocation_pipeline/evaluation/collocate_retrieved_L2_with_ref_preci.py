from __future__ import annotations

from pathlib import Path
import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr

from tqdm import tqdm

# Prefer stdlib tomllib if available (Python 3.11+), otherwise fall back to toml
try:
    import tomllib as _toml  # type: ignore
except ModuleNotFoundError:  # Python <= 3.10
    import toml as _toml  # type: ignore

import AVHRR_collocation_pipeline.utils as utils
from AVHRR_collocation_pipeline.readers.AVHRR_reader import read_AVHRR_orbit_to_df

from AVHRR_collocation_pipeline.readers.IMERG_reference import load_IMERG_reference
from AVHRR_collocation_pipeline.readers.ERA5_reference import load_ERA5_reference

from AVHRR_collocation_pipeline.readers.IMERG_reader import collocate_IMERG_precip
from AVHRR_collocation_pipeline.readers.ERA5_reader import collocate_ERA5_precip


# -----------------------------
# Worker-global caches (per process)
# -----------------------------
_CFG = None
_GRID = None  # (x_vec, y_vec, x, y)
_ERA5_META_BY_YEAR = None
_IMERG_META = None


def _init_worker(cfg: dict):
    """
    Runs ONCE per worker process.
    Loads heavy references + builds WGS grid once, then cached in that worker.
    """
    global _CFG, _GRID, _ERA5_META_BY_YEAR, _IMERG_META

    _CFG = cfg

    grid_res = float(cfg["grid"]["resolution_deg"])
    x_vec, y_vec, x, y = utils.build_test_grid(grid_res)
    _GRID = (x_vec, y_vec, x, y)

    # --- ERA5 (required) ---
    era5_dir = cfg["paths"]["era5_dir"]
    print(f"[Worker {os.getpid()}] Loading ERA5 reference from: {era5_dir}", flush=True)
    _ERA5_META_BY_YEAR = load_ERA5_reference(era5_dir)

    # --- IMERG (optional) ---
    use_imerg = bool((cfg.get("flags", {}) or {}).get("use_imerg", False))
    if use_imerg:
        imerg_dir = cfg["paths"]["imerg_dir"]
        print(f"[Worker {os.getpid()}] Loading IMERG reference from: {imerg_dir}", flush=True)
        _IMERG_META = load_IMERG_reference(imerg_dir)
    else:
        _IMERG_META = None
        print(f"[Worker {os.getpid()}] IMERG disabled (flags.use_imerg=false).", flush=True)


def _df_to_wgs_grids(
    df,
    x_vec,
    y_vec,
    x,
    y,
    varnames: list[str],
) -> dict[str, np.ndarray]:
    grids: dict[str, np.ndarray] = {}
    for v in varnames:
        if v not in df.columns:
            # print(f"[WARN] '{v}' not in df columns. Skipping grid.", flush=True)
            continue
        grids[v] = utils.df2grid(df, v, x_vec=x_vec, y_vec=y_vec, x=x, y=y).astype("float32")
    return grids


def _save_grids_to_nc(
    out_nc: Path,
    x_vec,
    y_vec,
    grids: dict[str, np.ndarray],
    attrs: dict,
):
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        coords={
            "y": ("y", y_vec.astype("float32")),
            "x": ("x", x_vec.astype("float32")),
        }
    )

    for name, arr in grids.items():
        if arr.ndim != 2:
            raise ValueError(f"Grid '{name}' is not 2D: shape={arr.shape}")
        ds[name] = (("y", "x"), arr)
        ds[name].attrs["coordinates"] = "y x"

    ds.attrs.update(attrs)

    enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_nc, format="NETCDF4", encoding=enc)


def process_one_orbit(l2_file_str: str) -> tuple[str, bool, str]:
    """
    Process ONE L2 orbit file.

    Returns: (orbit_tag, ok, message)
    """
    global _CFG, _GRID, _ERA5_META_BY_YEAR, _IMERG_META

    if _CFG is None or _GRID is None or _ERA5_META_BY_YEAR is None:
        return ("<unknown>", False, "Worker not initialized: missing cfg/grid/ERA5 meta.")

    l2_file = Path(l2_file_str)
    orbit_tag = l2_file.stem.replace("_L2", "")

    try:
        x_vec, y_vec, x, y = _GRID

        lat_thresh_nh = float(_CFG["grid"].get("lat_thresh_nh", 55.0))
        lat_thresh_sh = float(_CFG["grid"].get("lat_thresh_sh", -55.0))

        avh_vars = list(_CFG["vars"]["avh_vars"])
        grid_vars = list(_CFG["vars"]["grid_vars"])

        # ERA5 settings
        era5_var = _CFG["vars"].get("era5_var", "tp")
        era5_outcol = _CFG["vars"].get("era5_outcol", f"ERA5_{era5_var}")
        era5_scale = float(_CFG["vars"].get("era5_scale", 1000.0))

        # Output
        out_dir = Path(_CFG["paths"]["out_dir"])
        out_nc = out_dir / f"{orbit_tag}_collocated_wgs.nc"

        # 1) orbit -> df
        df = read_AVHRR_orbit_to_df(
            str(l2_file),
            x_vec=x_vec,
            y_vec=y_vec,
            x=x,
            y=y,
            avh_vars=avh_vars,
            lat_thresh_N_hemisphere=lat_thresh_nh,
            lat_thresh_S_hemisphere=lat_thresh_sh,
        )

        # 2) add time columns (scan_hour_unix / scan_halfhour_unix, scan_date, ...)
        df = utils.add_time_columns(df)

        # 3) IMERG (optional)
        if _IMERG_META is not None:
            df = collocate_IMERG_precip(df, _IMERG_META)

        # 4) ERA5 (required)
        df = collocate_ERA5_precip(
            df,
            _ERA5_META_BY_YEAR,
            varname=era5_var,
            out_col=era5_outcol,
            scale=era5_scale,
        )

        # 5) grid to WGS
        grids = _df_to_wgs_grids(df, x_vec, y_vec, x, y, grid_vars)

        # 6) save
        attrs = {
            "source_l2_file": str(l2_file),
            "grid_resolution_deg": float(_CFG["grid"]["resolution_deg"]),
            "lat_thresh_nh": lat_thresh_nh,
            "lat_thresh_sh": lat_thresh_sh,
            "era5_var": str(era5_var),
            "comment": "WGS grids from L2 orbit + collocated ERA5 (hourly) and optional IMERG (half-hourly).",
        }
        _save_grids_to_nc(out_nc, x_vec, y_vec, grids, attrs)

        return orbit_tag, True, str(out_nc)

    except Exception as e:
        tb = traceback.format_exc()
        return orbit_tag, False, f"{e}\n{tb}"


def list_l2_files(l2_dir: Path, pattern: str) -> list[Path]:
    return sorted(l2_dir.glob(pattern))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--pattern", type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = _toml.load(f)

    l2_dir = Path(cfg["paths"]["l2_dir"])
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # pattern: CLI overrides config, else config, else default
    pattern = args.pattern or cfg.get("paths", {}).get("pattern", "*_L2.nc")

    files = list_l2_files(l2_dir, pattern)
    if not files:
        raise SystemExit(f"No files matched {pattern} in {l2_dir}")

    # print(f"[INFO] Found {len(files)} files. Using n_workers={args.n_workers}", flush=True)

    ok = 0
    bad = 0

    with ProcessPoolExecutor(
        max_workers=args.n_workers,
        initializer=_init_worker,
        initargs=(cfg,),
    ) as ex:
        futs = [ex.submit(process_one_orbit, str(p)) for p in files]

        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc="Processing orbits",
            unit="orbit",
        ):
            orbit_tag, success, msg = fut.result()
            if success:
                ok += 1
                # print(f"[OK]   {orbit_tag} -> {msg}", flush=True)
            else:
                bad += 1
                # print(f"[FAIL] {orbit_tag}\n{msg}", flush=True)

    print(f"\n[SUMMARY] OK={ok}, FAIL={bad}, TOTAL={len(files)}", flush=True)


if __name__ == "__main__":
    main()