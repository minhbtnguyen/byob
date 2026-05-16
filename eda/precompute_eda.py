"""
Run once locally to precompute EDA results and save to data/eda/.
Usage: python eda/precompute_eda.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

_DATASET_ROOT = (
    Path.home()
    / ".cache/kagglehub/datasets/arashnic/microsoft-geolife-gps-trajectory-dataset"
    / "versions/1/Geolife Trajectories 1.3/Data"
)

_TARGET_MODES = ["walk", "bike", "bus", "car", "subway", "taxi"]
_EMISSION_FACTORS = {
    "car": 170,
    "taxi": 170,
    "bus": 89,
    "subway": 41,
    "bike": 0,
    "walk": 0,
}
_OUT = Path(__file__).parent.parent / "data" / "eda"


def _parse_plt_full(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        skiprows=6,
        header=None,
        names=["lat", "lon", "_zero", "altitude", "days", "date", "time"],
    )
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y-%m-%d %H:%M:%S"
    )
    df["user"] = filepath.parts[-3]
    return df[["user", "lat", "lon", "altitude", "datetime"]]


def _parse_labels(user_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        user_dir / "labels.txt",
        sep="\t",
        skiprows=1,
        header=None,
        names=["start", "end", "mode"],
    )
    df["start"] = pd.to_datetime(df["start"], format="%Y/%m/%d %H:%M:%S")
    df["end"] = pd.to_datetime(df["end"], format="%Y/%m/%d %H:%M:%S")
    df["duration_min"] = (df["end"] - df["start"]).dt.total_seconds() / 60
    df["mode"] = df["mode"].str.lower().str.strip()
    df["user"] = user_dir.name
    return df


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_label_coverage(root: Path) -> pd.DataFrame:
    labeled = [
        u for u in sorted(root.iterdir()) if u.is_dir() and (u / "labels.txt").exists()
    ]
    rows = []
    for i, user_dir in enumerate(labeled):
        print(f"  label coverage {i+1}/{len(labeled)}: {user_dir.name}")
        labels = _parse_labels(user_dir)
        labeled_s = (labels["end"] - labels["start"]).dt.total_seconds().sum()
        all_times = []
        for f in sorted((user_dir / "Trajectory").glob("*.plt")):
            all_times.extend(_parse_plt_full(f)["datetime"].tolist())
        if not all_times:
            continue
        total_s = (max(all_times) - min(all_times)).total_seconds()
        rows.append(
            {
                "user": user_dir.name,
                "pct_labeled": 100 * labeled_s / total_s if total_s > 0 else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("pct_labeled")


def load_all_labels(root: Path) -> pd.DataFrame:
    labeled = [
        u for u in sorted(root.iterdir()) if u.is_dir() and (u / "labels.txt").exists()
    ]
    dfs = []
    for i, u in enumerate(labeled):
        print(f"  loading labels {i+1}/{len(labeled)}: {u.name}")
        dfs.append(_parse_labels(u))
    return pd.concat(dfs, ignore_index=True)


def compute_speed_by_mode(root: Path, all_labels: pd.DataFrame) -> dict:
    labeled = [
        u for u in sorted(root.iterdir()) if u.is_dir() and (u / "labels.txt").exists()
    ]
    speed_by_mode: dict = {m: [] for m in _TARGET_MODES}
    for i, user_dir in enumerate(labeled):
        print(f"  speed profiles {i+1}/{len(labeled)}: {user_dir.name}")
        labels = all_labels[all_labels["user"] == user_dir.name].copy()
        labels = labels[labels["mode"].isin(_TARGET_MODES)]
        if labels.empty:
            continue
        plts = sorted((user_dir / "Trajectory").glob("*.plt"))
        traj = pd.concat(
            [_parse_plt_full(f) for f in plts], ignore_index=True
        ).sort_values("datetime")
        dist = np.zeros(len(traj))
        dist[1:] = _haversine_m(
            traj["lat"].values[:-1],
            traj["lon"].values[:-1],
            traj["lat"].values[1:],
            traj["lon"].values[1:],
        )
        dt_s = traj["datetime"].diff().dt.total_seconds().fillna(0).values
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dt_s > 0, (dist / dt_s) * 3.6, 0)
        traj = traj.copy()
        traj["speed_kmh"] = speed
        traj = traj[traj["speed_kmh"] < 250]
        for _, row in labels.iterrows():
            mask = (traj["datetime"] >= row["start"]) & (traj["datetime"] <= row["end"])
            pts = traj.loc[mask, "speed_kmh"]
            if len(pts) > 2:
                speed_by_mode[row["mode"]].extend(pts.tolist())
    return speed_by_mode


def compute_emissions(all_labels: pd.DataFrame, speed_by_mode: dict) -> pd.DataFrame:
    median_speed = {
        mode: float(np.median(speeds)) if speeds else 30.0
        for mode, speeds in speed_by_mode.items()
    }
    df = all_labels[all_labels["mode"].isin(_EMISSION_FACTORS)].copy()
    df["speed_kmh"] = df["mode"].map(median_speed).fillna(30.0)
    df["dist_km"] = (df["duration_min"] / 60) * df["speed_kmh"]
    df["co2_g"] = df["dist_km"] * df["mode"].map(_EMISSION_FACTORS)
    return df[["user", "mode", "dist_km", "co2_g"]].reset_index(drop=True)


def main():
    if not _DATASET_ROOT.exists():
        print(f"Dataset not found at {_DATASET_ROOT}")
        print("Run the first cell of eda/analysis.ipynb to download it.")
        return

    _OUT.mkdir(parents=True, exist_ok=True)
    print("Starting precomputation...")

    print("\n[1/4] Label coverage")
    if (_OUT / "label_coverage.parquet").exists():
        print("  Skipping — already exists")
        cov_df = pd.read_parquet(_OUT / "label_coverage.parquet")
    else:
        cov_df = compute_label_coverage(_DATASET_ROOT)
        cov_df.to_parquet(_OUT / "label_coverage.parquet", index=False)
        print(f"  Saved {len(cov_df)} rows")

    print("\n[2/4] All labels")
    if (_OUT / "all_labels.parquet").exists():
        print("  Skipping — already exists")
        all_labels = pd.read_parquet(_OUT / "all_labels.parquet")
    else:
        all_labels = load_all_labels(_DATASET_ROOT)
        all_labels.to_parquet(_OUT / "all_labels.parquet", index=False)
        print(f"  Saved {len(all_labels)} rows")

    print("\n[3/4] Speed by mode")
    if (_OUT / "speed_by_mode.json").exists():
        print("  Skipping — already exists")
        with open(_OUT / "speed_by_mode.json") as f:
            sbm = json.load(f)
    else:
        sbm = compute_speed_by_mode(_DATASET_ROOT, all_labels)
        with open(_OUT / "speed_by_mode.json", "w") as f:
            json.dump(sbm, f)
        print(f"  Saved {sum(len(v) for v in sbm.values()):,} speed samples")

    print("\n[3b/4] Speed histograms (precomputed bins)")
    if (_OUT / "speed_histograms.json").exists():
        print("  Skipping — already exists")
    else:
        histograms = {}
        for mode, speeds in sbm.items():
            arr = np.array(speeds)
            clipped = arr[(arr >= 0) & (arr <= 120)]
            counts, bin_edges = np.histogram(clipped, bins=50, density=True)
            histograms[mode] = {
                "counts": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
                "median": float(np.median(arr)) if len(arr) > 0 else 0.0,
                "n": int(len(arr)),
            }
        with open(_OUT / "speed_histograms.json", "w") as f:
            json.dump(histograms, f)
        print(f"  Saved histograms for {len(histograms)} modes")

    print("\n[3c/4] Trajectory samples (5 per user for map viewer)")
    if (_OUT / "trajectory_samples.parquet").exists():
        print("  Skipping — already exists")
    else:
        users = sorted([u for u in _DATASET_ROOT.iterdir() if u.is_dir()])
        dfs = []
        for user_dir in users:
            plts = sorted((user_dir / "Trajectory").glob("*.plt"))[:5]
            for plt_file in plts:
                try:
                    df = pd.read_csv(
                        plt_file, skiprows=6, header=None,
                        names=["lat", "lon", "zero", "altitude_ft", "days", "date", "time"],
                    )
                    df["datetime"] = pd.to_datetime(
                        df["date"] + " " + df["time"], format="%Y-%m-%d %H:%M:%S"
                    )
                    df["user"] = user_dir.name
                    df["filename"] = plt_file.name
                    dfs.append(df[["user", "filename", "datetime", "lat", "lon", "altitude_ft"]])
                except Exception:
                    continue
        traj_df = pd.concat(dfs, ignore_index=True)
        traj_df.to_parquet(_OUT / "trajectory_samples.parquet", index=False)
        print(f"  Saved {len(traj_df):,} GPS points across {traj_df['user'].nunique()} users")

    print("\n[4/4] Emissions")
    if (_OUT / "emissions.parquet").exists():
        print("  Skipping — already exists")
    else:
        em_df = compute_emissions(all_labels, sbm)
        em_df.to_parquet(_OUT / "emissions.parquet", index=False)
        print(f"  Saved {len(em_df)} rows")

    print("\nDone. Files written to data/eda/")


if __name__ == "__main__":
    main()
