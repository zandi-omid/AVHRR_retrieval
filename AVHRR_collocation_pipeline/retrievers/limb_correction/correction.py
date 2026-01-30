# AVHRR_collocation_pipeline/retrievers/limb_correction/correction.py

from __future__ import annotations

import datetime
import re
from typing import Tuple, Optional, Dict, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from .lut_loader import LimbCorrAssets


# -----------------------------
# Helpers for orbit_name -> season
# -----------------------------
def extract_year_and_doy(file_name: str) -> Tuple[int, int]:
    """
    Parse e.g. '...D19354...' => year=2019, doy=354
    """
    parts = file_name.split(".")
    d_parts = [part for part in parts if part.startswith("D")]
    if len(d_parts) == 0:
        raise ValueError(f"orbit_name {file_name!r} missing 'DYYDDD' token.")
    year_prefix = d_parts[0][1:3]
    day_of_year = int(d_parts[0][3:6])
    year = 1900 + int(year_prefix) if int(year_prefix) >= 98 else 2000 + int(year_prefix)
    return year, day_of_year


def calculate_month(doy: int, year: int) -> int:
    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
    return dt.month


def find_season(month: int, hemisphere: str) -> str:
    """
    hemisphere: "Southern" or "Northern"
    """
    if hemisphere == "Southern":
        season_month_south = {
            12: "Summer", 1: "Summer", 2: "Summer",
            3: "Autumn", 4: "Autumn", 5: "Autumn",
            6: "Winter", 7: "Winter", 8: "Winter",
            9: "Spring", 10: "Spring", 11: "Spring",
        }
        return season_month_south[month]

    if hemisphere == "Northern":
        season_month_north = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn",
        }
        return season_month_north[month]

    raise ValueError(f"Invalid hemisphere: {hemisphere!r}")


# -----------------------------
# Latitude-bin parsing / formatting
# -----------------------------
def parse_lat_window(lat) -> Tuple[int, int]:
    if isinstance(lat, tuple):
        return tuple(sorted(map(int, lat)))

    s = str(lat).strip()
    nums = re.findall(r"-?\d+", s)
    if len(nums) != 2:
        raise ValueError(f"Unrecognized latitude_bin format: {lat}")

    a, b = map(int, nums)

    # Fix common "61-75" case where regex gives [61, -75]
    if ("-" in s and "--" not in s and not s.strip().startswith("-") and a > 0 and b < 0):
        b = abs(b)

    return tuple(sorted((a, b)))


def lat_window_to_bin(lat_window: Tuple[int, int]) -> str:
    lo, hi = lat_window
    return f"{lo}-{hi}"


# -----------------------------
# LUT preprocessing
# -----------------------------
def preprocess_lut_fast(LUT: pd.DataFrame):
    """
    lut[lat_bin][beam][surf_type][obs_temp] = corr_coeff
    """
    lut_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in LUT.itertuples(index=False):
        lut_dict[row.latitude_bin][row.beam_position][row.surface_type][row.original_tb] = row.corr_coeff
    return lut_dict


def get_correction_fast(latwind: str, beam: int, surf_type: int, obs_temp: float, lut_dict) -> Optional[float]:
    try:
        beam_dict = lut_dict.get(str(latwind), {})
        surf_dict = beam_dict.get(int(beam), {}).get(int(surf_type), {})
        if not surf_dict:
            return None
        temp_keys = np.array(list(surf_dict.keys()))
        if temp_keys.size == 0:
            return None
        temp_key = temp_keys[np.abs(temp_keys - obs_temp).argmin()]
        return float(surf_dict[temp_key])
    except Exception:
        return None


# -----------------------------
# Decision logic (copied/kept consistent with your friend's rules)
# -----------------------------
def get_custom_surface_type_mapping(hemisphere: str, season: str, lat_range: Tuple[int, int]) -> Optional[Dict[int, str]]:
    seesns = ["Summer", "Autumn", "Winter", "Spring"]

    if hemisphere == "SH":
        if (season in seesns) and (lat_range == (-75, -61)):
            return {0: "water", 2: "snow-covered land", 3: "ice"}
        elif (season in ["Summer", "Spring"]) and (lat_range == (-61, -53)):
            return {0: "water", 1: "snow-free land"}
        elif (season in ["Autumn", "Winter"]) and (lat_range == (-61, -53)):
            return {0: "water", 3: "ice"}
        elif (season in ["Summer", "Spring"]) and (lat_range == (-53, -45)):
            return {0: "water"}
        elif (season in ["Winter", "Autumn"]) and (lat_range == (-53, -45)):
            return {0: "water", 3: "ice"}

    if hemisphere == "NH":
        if (season in ["Winter", "Autumn"]) and (lat_range == (61, 75)):
            return {0: "water", 2: "snow-covered land", 3: "ice"}
        elif (season in ["Spring", "Summer"]) and (lat_range == (61, 75)):
            return {0: "water", 1: "snow-free land"}
        elif (season in ["Spring", "Summer"]) and (lat_range == (53, 61)):
            return {0: "water", 1: "snow-free land"}
        elif (season in ["Winter", "Autumn"]) and (lat_range == (53, 61)):
            return {0: "water", 2: "snow-covered land", 3: "ice"}
        elif (season in ["Spring", "Summer"]) and (lat_range == (45, 53)):
            return {0: "water", 1: "snow-free land"}
        elif (season in ["Winter", "Autumn"]) and (lat_range == (45, 53)):
            return {0: "water", 2: "snow-covered land", 3: "ice"}

    return None


def decide_correction_mode(hemisphere: str, season: str, lat_range: Tuple[int, int], surface_name: str) -> str:
    valid_map = get_custom_surface_type_mapping(hemisphere, season, lat_range) or {}

    if surface_name not in valid_map.values():
        return "GLOBAL_CURVE"

    if hemisphere == "SH":
        if surface_name == "water":
            return "LUT"
        if surface_name in ["ice", "snow-covered land"]:
            return "SURFACE_CURVE"
        return "GLOBAL_CURVE"

    if hemisphere == "NH":
        if surface_name == "water":
            return "LUT"

        if surface_name == "snow-covered land":
            if season == "Winter" and lat_range in [(53, 61), (61, 75)]:
                return "LUT"
            return "SURFACE_CURVE"

        if surface_name == "snow-free land":
            if season in ["Spring", "Summer"]:
                return "LUT"
            return "SURFACE_CURVE"

        if surface_name == "ice":
            return "SURFACE_CURVE"

    return "GLOBAL_CURVE"


def surface_curve_available(curve_lib: Dict[str, Any], var: str, hemisphere: str, season: str, lat_bin: str, surface_code: int) -> bool:
    try:
        return int(surface_code) in curve_lib[var][hemisphere][season][lat_bin]
    except Exception:
        return False


def eval_geometry_curve(beam_position: np.ndarray, coeffs) -> np.ndarray:
    # coeffs is polynomial coefficients; x = beam - NADIR_CENTER is already embedded in coeffs in your pickles
    return np.polyval(coeffs, beam_position - 204)


# -----------------------------
# Valid-index selection
# -----------------------------
def get_valid_indices_and_data(
    lat_window_str: str,
    lat_range: Tuple[int, int],
    surface_type: np.ndarray,
    brightness_temp: np.ndarray,
    lut_df: pd.DataFrame,
    lats: np.ndarray,
    cloud_probs_msk: np.ndarray,
    limb_beam_positions: np.ndarray,
):
    max_lat, min_lat = int(max(lat_range)), int(min(lat_range))
    lat_msk = (lats > min_lat) & (lats <= max_lat)

    valid_surface_types = set(
        lut_df[lut_df["latitude_bin"] == str(lat_window_str)]["surface_type"].unique()
    )

    valid_mask = (
        lat_msk
        & np.isin(surface_type, list(valid_surface_types))
        & (~np.isnan(cloud_probs_msk))
        & (~np.isnan(brightness_temp))
    )

    valid_indices = np.argwhere(valid_mask)
    if valid_indices.size == 0:
        return None, None, None, None

    # Keep only limb beams
    good = np.isin(valid_indices[:, 1], limb_beam_positions)
    valid_indices = valid_indices[good]
    if valid_indices.size == 0:
        return None, None, None, None

    i_idx = valid_indices[:, 0]
    j_idx = valid_indices[:, 1]
    temp_tb = brightness_temp[i_idx, j_idx]
    surf_vals = surface_type[i_idx, j_idx]
    return temp_tb, surf_vals, i_idx, j_idx


# -----------------------------
# Correction core for one channel
# -----------------------------
def apply_channel_correction_grouped(
    *,
    assets: LimbCorrAssets,
    var: str,                        # "temp_11" or "temp_12"
    temp_tb: np.ndarray,
    surface_type_val: np.ndarray,
    i_indices: np.ndarray,
    j_indices: np.ndarray,
    lat_bin_tuple: Tuple[int, int],  # e.g. (61, 75)
    lat_window: str,                 # e.g. "61-75"
    hemisphere: str,                 # "NH"/"SH"
    season: str,                     # e.g. "Winter"
    lut_dict,
    lut_df: pd.DataFrame,
    corrected_tb: np.ndarray,
):
    surface_type_mapping = assets.surface_type_mapping
    global_curve = assets.global_curve
    curve_lib = assets.curve_lib

    for surface_code in np.unique(surface_type_val):
        surface_code = int(surface_code)
        if surface_code not in surface_type_mapping:
            continue

        surface_name = surface_type_mapping[surface_code]

        mask = surface_type_val == surface_code
        if not np.any(mask):
            continue

        obs_tb = temp_tb[mask]
        beams  = j_indices[mask]
        i_idx  = i_indices[mask]
        j_idx  = j_indices[mask]

        mode = decide_correction_mode(
            hemisphere=hemisphere,
            season=season,
            lat_range=lat_bin_tuple,
            surface_name=surface_name,
        )

        if mode == "LUT":
            # Vectorized LUT lookup (returns None sometimes)
            corr = np.vectorize(get_correction_fast, otypes=[float])(
                lat_window, beams, surface_code, obs_tb, lut_dict
            )

            # If any missing, fall back to 1.0 (no correction) for those pixels
            corr = np.where(np.isfinite(corr), corr, 1.0)
            corrected = obs_tb * corr

        elif mode == "SURFACE_CURVE":
            use_surface = surface_curve_available(curve_lib, var, hemisphere, season, lat_window, surface_code)
            if use_surface:
                entry = curve_lib[var][hemisphere][season][lat_window][surface_code]
                coeffs = entry["coeffs"]
            else:
                entry = global_curve[var][hemisphere][season][lat_window]
                coeffs = entry["coeffs"]

            factors = eval_geometry_curve(beams, coeffs)
            corrected = obs_tb * factors

        elif mode == "GLOBAL_CURVE":
            entry = global_curve[var][hemisphere][season][lat_window]
            coeffs = entry["coeffs"]
            factors = eval_geometry_curve(beams, coeffs)
            corrected = obs_tb * factors

        else:
            corrected = obs_tb

        corrected_tb[i_idx, j_idx] = corrected


# -----------------------------
# Public API: correct swath dataset
# -----------------------------
def correct_dataset_vectorized(
    ds: xr.Dataset,
    *,
    orbit_name: str,
    assets: LimbCorrAssets,
    cloud_prob_threshold: float = 0.5,
    limb_half_width: int = 50,
) -> xr.Dataset:
    """
    Apply AVHRR IR TB limb correction to a swath xr.Dataset IN MEMORY.

    Requirements: ds must contain variables:
      - latitude (2D)
      - cloud_probability (2D)
      - land_class (2D int codes 0..3)
      - temp_11_0um_nom (2D)
      - temp_12_0um_nom (2D)

    Returns
    -------
    ds_out : xr.Dataset
        Same dataset with added variables:
          - temp_11_0um_nom_corrected
          - temp_12_0um_nom_corrected
    """
    required = ["latitude", "cloud_probability", "land_class", "temp_11_0um_nom", "temp_12_0um_nom"]
    missing = [v for v in required if v not in ds]
    if missing:
        raise KeyError(f"Dataset missing required variables for limb correction: {missing}")

    # Determine season using orbit_name
    year, doy = extract_year_and_doy(orbit_name)
    # leap-year adjustment consistent with your friend's logic
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    if is_leap and doy > 59:
        doy -= 1
    month = calculate_month(doy, year)
    sh_season = find_season(month, "Southern")
    nh_season = {"Summer": "Winter", "Autumn": "Spring", "Winter": "Summer", "Spring": "Autumn"}[sh_season]

    # Load LUT frames for seasons, concatenate NH+SH for each channel
    all_lut = assets.all_lut
    if "temp_11" not in all_lut or "temp_12" not in all_lut:
        raise KeyError("assets.all_lut missing temp_11 or temp_12 keys. Check LUT filenames / loader.")

    luts_11_nh = all_lut["temp_11"].get("NH", {})
    luts_11_sh = all_lut["temp_11"].get("SH", {})
    luts_12_nh = all_lut["temp_12"].get("NH", {})
    luts_12_sh = all_lut["temp_12"].get("SH", {})

    if nh_season not in luts_11_nh or sh_season not in luts_11_sh:
        raise KeyError(f"Missing temp_11 LUT season: NH={nh_season}, SH={sh_season}")
    if nh_season not in luts_12_nh or sh_season not in luts_12_sh:
        raise KeyError(f"Missing temp_12 LUT season: NH={nh_season}, SH={sh_season}")

    lut_11 = pd.concat([luts_11_nh[nh_season], luts_11_sh[sh_season]], ignore_index=True)
    lut_12 = pd.concat([luts_12_nh[nh_season], luts_12_sh[sh_season]], ignore_index=True)

    # Available lat windows (strings in LUT like "61-75" or "(61, 75)")
    lat_bins = [parse_lat_window(lat) for lat in lut_11["latitude_bin"].unique()]
    # Convert to "lo-hi" strings used by lut_dict keys
    lat_windows = [lat_window_to_bin(lb) for lb in lat_bins]

    # Precompute LUT dicts
    lut_11_dict = preprocess_lut_fast(lut_11)
    lut_12_dict = preprocess_lut_fast(lut_12)

    # Inputs
    lats = ds["latitude"].values
    cloud_probs = ds["cloud_probability"].values
    cloud_probs_msk = np.where(cloud_probs >= cloud_prob_threshold, cloud_probs, np.nan)

    surface_type = ds["land_class"].values
    tb11 = ds["temp_11_0um_nom"].values
    tb12 = ds["temp_12_0um_nom"].values

    corrected_11 = np.array(tb11, copy=True)
    corrected_12 = np.array(tb12, copy=True)

    # Beam geometry: 409 beams, nadir is median, limb beams are outside +/- limb_half_width
    beam_positions = np.arange(tb11.shape[1], dtype=int)  # assumes dim1 is beam
    nadir = int(np.median(beam_positions))
    ref_beams = np.arange(nadir - limb_half_width, nadir + limb_half_width)
    limb_beams = beam_positions[~np.isin(beam_positions, ref_beams)]

    # Iterate lat windows and apply correction for each channel
    for lat_bin_tuple, lat_window in zip(lat_bins, lat_windows):
        # Infer hemisphere from lat bin tuple (same as your friend’s logic)
        if int(lat_bin_tuple[1]) < 0:
            hemi = "SH"
            season = sh_season
        elif int(lat_bin_tuple[0]) > 0:
            hemi = "NH"
            season = nh_season
        else:
            # equator-crossing bin is ignored
            continue

        # ---- 11um ----
        t11, s11, i11, j11 = get_valid_indices_and_data(
            lat_window, lat_bin_tuple, surface_type, tb11, lut_11, lats, cloud_probs_msk, limb_beams
        )
        if t11 is not None:
            apply_channel_correction_grouped(
                assets=assets,
                var="temp_11",
                temp_tb=t11,
                surface_type_val=s11,
                i_indices=i11,
                j_indices=j11,
                lat_bin_tuple=lat_bin_tuple,
                lat_window=lat_window,
                hemisphere=hemi,
                season=season,
                lut_dict=lut_11_dict,
                lut_df=lut_11,
                corrected_tb=corrected_11,
            )

        # ---- 12um ----
        t12, s12, i12, j12 = get_valid_indices_and_data(
            lat_window, lat_bin_tuple, surface_type, tb12, lut_12, lats, cloud_probs_msk, limb_beams
        )
        if t12 is not None:
            apply_channel_correction_grouped(
                assets=assets,
                var="temp_12",
                temp_tb=t12,
                surface_type_val=s12,
                i_indices=i12,
                j_indices=j12,
                lat_bin_tuple=lat_bin_tuple,
                lat_window=lat_window,
                hemisphere=hemi,
                season=season,
                lut_dict=lut_12_dict,
                lut_df=lut_12,
                corrected_tb=corrected_12,
            )

    # Return dataset with new vars (keep dims identical to originals)
    ds_out = ds.copy()
    ds_out["temp_11_0um_nom_corrected"] = (ds["temp_11_0um_nom"].dims, corrected_11)
    ds_out["temp_12_0um_nom_corrected"] = (ds["temp_12_0um_nom"].dims, corrected_12)

    return ds_out