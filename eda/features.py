"""
features.py
Feature engineering for Geolife GPS trajectories.

Produces a flat DataFrame with one row per labeled 30s window:
  columns = [user, window_start, window_end, mode, <9 heuristic features>]

Usage:
    from features import build_feature_dataset
    df = build_feature_dataset(DATA_DIR)
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_SEC       = 30    # window length in seconds
STRIDE_SEC       = 15    # stride between window starts (50% overlap)
MIN_POINTS       = 5     # minimum GPS points per window to be usable
SPEED_LIMIT_KMH  = 250   # points above this are GPS anomalies → discard window
MIN_WINDOWS_USER = 50    # skip users with fewer than this many labeled windows
MAX_WINDOWS_USER = 5000  # cap per user to prevent heavy users dominating

# Core 4 modes — taxi merged into car, subway/boat/airplane dropped (too rare)
TARGET_MODES = {"walk", "bike", "bus", "car"}
MODE_MERGE   = {"taxi": "car", "run": "walk", "subway": None, "boat": None, "airplane": None}

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_plt(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(
        filepath, skiprows=6, header=None,
        names=["lat", "lon", "_zero", "altitude", "days", "date", "time"],
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    return df[["lat", "lon", "altitude", "datetime"]].copy()


def parse_labels(user_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        user_dir / "labels.txt", sep="\t", skiprows=1, header=None,
        names=["start", "end", "mode"],
    )
    df["start"] = pd.to_datetime(df["start"])
    df["end"]   = pd.to_datetime(df["end"])
    df["mode"] = df["mode"].str.lower().str.strip().replace(MODE_MERGE)
    # Drop rows where mode mapped to None (subway, boat, airplane)
    df = df[df["mode"].notna() & df["mode"].isin(TARGET_MODES)]
    return df.reset_index(drop=True)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised haversine distance in metres between consecutive point pairs."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_bearing(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Compass bearing in degrees [0, 360) between consecutive point pairs."""
    dlon   = np.radians(lon2 - lon1)
    lat1r  = np.radians(lat1)
    lat2r  = np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y)) % 360


# ── Per-window feature extraction ─────────────────────────────────────────────

def extract_features(window: pd.DataFrame) -> Optional[dict]:
    """
    Compute heuristic features for a single GPS window.

    Returns None if the window is too sparse or contains a GPS anomaly.
    """
    if len(window) < MIN_POINTS:
        return None

    lats  = window["lat"].values
    lons  = window["lon"].values
    times = window["datetime"].values.astype("datetime64[s]").astype(np.float64)

    # Distances and time deltas between consecutive points
    dist_m = np.zeros(len(window))
    dt_s   = np.zeros(len(window))
    dist_m[1:] = haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
    dt_s[1:]   = np.diff(times)

    # Speed (km/h) — guard against zero dt
    with np.errstate(divide="ignore", invalid="ignore"):
        speed_kmh = np.where(dt_s > 0, (dist_m / dt_s) * 3.6, 0.0)

    # Discard window if any GPS anomaly or large internal time gap
    if np.any(speed_kmh > SPEED_LIMIT_KMH):
        return None
    if np.any(dt_s > 15):  # gap >15s in a 30s window = unreliable features
        return None

    # Acceleration (Δspeed / Δtime, km/h per second)
    accel = np.zeros_like(speed_kmh)
    accel[1:] = np.where(dt_s[1:] > 0, np.diff(speed_kmh) / dt_s[1:], 0.0)

    # Jerk (Δaccel / Δtime — measures motion roughness)
    jerk = np.zeros_like(accel)
    jerk[1:] = np.where(dt_s[1:] > 0, np.diff(accel) / dt_s[1:], 0.0)

    # Bearing variance (direction change — high for walk/bus, low for subway)
    if len(lats) > 2:
        bearings     = compute_bearing(lats[:-1], lons[:-1], lats[1:], lons[1:])
        bearing_var  = float(np.var(bearings))
    else:
        bearing_var  = 0.0

    return {
        "speed_mean":        float(np.mean(speed_kmh)),
        "speed_max":         float(np.max(speed_kmh)),
        "speed_std":         float(np.std(speed_kmh)),
        "accel_mean":        float(np.mean(np.abs(accel))),
        "accel_std":         float(np.std(accel)),
        "jerk_mean":         float(np.mean(np.abs(jerk))),
        "stop_ratio":        float(np.mean(speed_kmh < 1.0)),
        "bearing_variance":  bearing_var,
        "distance_total_m":  float(np.sum(dist_m)),
    }


# ── Label assignment ──────────────────────────────────────────────────────────

def assign_label(
    window_start: pd.Timestamp,
    window_end:   pd.Timestamp,
    labels:       pd.DataFrame,
    min_overlap:  float = 0.5,
) -> Optional[str]:
    """
    Return the mode label with the most overlap with this window,
    provided that overlap covers at least min_overlap (default 50%) of the window.
    Returns None if no label meets the threshold.
    """
    window_dur = (window_end - window_start).total_seconds()

    # Compute overlap duration with each label interval
    overlap_start = labels["start"].clip(lower=window_start)
    overlap_end   = labels["end"].clip(upper=window_end)
    overlap_sec   = (overlap_end - overlap_start).dt.total_seconds().clip(lower=0)

    best_idx = overlap_sec.idxmax()
    if overlap_sec[best_idx] / window_dur >= min_overlap:
        return labels.loc[best_idx, "mode"]
    return None


# ── Per-user processing ───────────────────────────────────────────────────────

def load_trajectory(user_dir: Path) -> pd.DataFrame:
    """Load and merge all .plt files for a user into one time-sorted DataFrame."""
    plts = sorted((user_dir / "Trajectory").glob("*.plt"))
    if not plts:
        return pd.DataFrame()
    traj = pd.concat([parse_plt(f) for f in plts], ignore_index=True)
    return traj.sort_values("datetime").reset_index(drop=True)


def process_user(user_dir: Path) -> tuple[list[dict], str]:
    """
    Slide windows over labeled intervals only (not the full recording span).
    Uses numpy searchsorted for O(log n) window slicing instead of full scans.

    Returns (rows, skip_reason) — skip_reason is non-empty if user was skipped.
    """
    traj = load_trajectory(user_dir)
    if traj.empty:
        return [], "no trajectory"

    labels = parse_labels(user_dir)
    if labels.empty:
        return [], "no labels in target modes"

    # Pre-compute unix timestamps as numpy array for fast searchsorted
    ts = traj["datetime"].values.astype("datetime64[s]").astype(np.int64)

    window_sec = WINDOW_SEC
    stride_sec = STRIDE_SEC
    rows = []

    for _, label_row in labels.iterrows():
        if len(rows) >= MAX_WINDOWS_USER:
            break

        t = label_row["start"]
        label_end = label_row["end"]

        while t + pd.Timedelta(seconds=window_sec) <= label_end:
            if len(rows) >= MAX_WINDOWS_USER:
                break

            w_end = t + pd.Timedelta(seconds=window_sec)

            # O(log n) slice using searchsorted on unix timestamps
            t_int     = int(t.timestamp())
            w_end_int = int(w_end.timestamp())
            lo = np.searchsorted(ts, t_int,     side="left")
            hi = np.searchsorted(ts, w_end_int, side="left")
            window = traj.iloc[lo:hi]

            # Mode is known — we're sliding within this label interval already
            mode = label_row["mode"]
            feats = extract_features(window)
            if feats is not None:
                rows.append({
                    "user":         user_dir.name,
                    "window_start": t,
                    "window_end":   w_end,
                    "mode":         mode,
                    **feats,
                })

            t += pd.Timedelta(seconds=stride_sec)

    if len(rows) < MIN_WINDOWS_USER:
        return [], f"only {len(rows)} windows (<{MIN_WINDOWS_USER} min)"

    return rows, ""


# ── Main entry point ──────────────────────────────────────────────────────────

# Default cache location relative to this file
_DEFAULT_CACHE = Path(__file__).parent.parent / "data" / "processed" / "features.parquet"


def build_feature_dataset(
    data_dir: Path,
    cache_path: Path = _DEFAULT_CACHE,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Process all labeled users and return a flat feature DataFrame.
    Results are cached to Parquet — subsequent calls load from disk instantly.

    Args:
        data_dir:      Path to the Geolife Data/ directory
        cache_path:    Where to save/load the cached Parquet file
        force_rebuild: If True, recompute even if cache exists
        verbose:       Print per-user window counts

    Returns:
        DataFrame — one row per labeled window, ready for LightGBM.
        Columns: user, window_start, window_end, mode, + 9 features
    """
    # Load from cache if available
    if cache_path.exists() and not force_rebuild:
        if verbose:
            print(f"Loading features from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        df["mode"] = pd.Categorical(df["mode"], categories=sorted(TARGET_MODES))
        if verbose:
            print(f"Loaded {len(df):,} windows")
        return df

    # Build from scratch — sequential with progress bar
    from tqdm import tqdm

    labeled_users = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / "labels.txt").exists()
    ])

    all_rows = []
    for user_dir in tqdm(labeled_users, desc="Processing users"):
        rows, skip_reason = process_user(user_dir)
        all_rows.extend(rows)
        if verbose:
            if skip_reason:
                tqdm.write(f"  {user_dir.name}: skipped ({skip_reason})")
            else:
                tqdm.write(f"  {user_dir.name}: {len(rows):>5} windows")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["mode"] = pd.Categorical(df["mode"], categories=sorted(TARGET_MODES))

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    if verbose:
        print(f"\nTotal windows : {len(df):,}")
        print(f"Mode breakdown:\n{df['mode'].value_counts().to_string()}")
        print(f"\nCached to: {cache_path}")

    return df
