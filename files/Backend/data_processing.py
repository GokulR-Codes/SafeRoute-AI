"""
Namma Safe BLR — Data Processing Module
Cleans crime data, clusters hotspots, computes density grids.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import json, os

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/bangalore_crime_dataset.csv")

# ─── Load & clean dataset ─────────────────────────────────────────────────────
def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop rows with missing coords
    df.dropna(subset=["latitude", "longitude", "crime_severity"], inplace=True)

    # Clip to Bangalore bounding box
    df = df[
        (df["latitude"].between(12.80, 13.15)) &
        (df["longitude"].between(77.45, 77.80))
    ].copy()

    # Normalize severity 0–1
    scaler = MinMaxScaler()
    df["severity_norm"] = scaler.fit_transform(df[["crime_severity"]])

    # Parse hour if not present
    if "hour" not in df.columns:
        df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour

    # Night flag
    df["is_night"] = ((df["hour"] >= 20) | (df["hour"] < 6)).astype(int)

    print(f"✅ Loaded {len(df)} clean crime records")
    return df


# ─── DBSCAN hotspot clustering ────────────────────────────────────────────────
def cluster_hotspots(df: pd.DataFrame, eps_km: float = 0.5, min_samples: int = 8):
    """
    DBSCAN in geographic space.  eps is in km (approx 0.009 deg ≈ 1 km).
    Returns df with 'cluster_id' and cluster summary dict.
    """
    coords = df[["latitude", "longitude"]].values
    eps_deg = eps_km / 111.0          # ~111 km per degree

    db = DBSCAN(eps=eps_deg, min_samples=min_samples, algorithm="ball_tree", metric="haversine")
    df = df.copy()
    df["cluster_id"] = db.fit_predict(np.radians(coords))

    # Summarise clusters (exclude noise = -1)
    valid = df[df["cluster_id"] >= 0]
    summary = (
        valid.groupby("cluster_id")
        .agg(
            lat=("latitude",      "mean"),
            lon=("longitude",     "mean"),
            count=("crime_type",  "count"),
            avg_severity=("severity_norm", "mean"),
        )
        .reset_index()
    )
    print(f"✅ Detected {len(summary)} crime hotspot clusters")
    return df, summary


# ─── Grid-based crime density map ─────────────────────────────────────────────
def compute_density_grid(df: pd.DataFrame, resolution: float = 0.01) -> dict:
    """
    Divides Bangalore into grid cells and computes crime density per cell.
    Returns list of {lat, lon, density} suitable for heatmap rendering.
    """
    df = df.copy()
    df["grid_lat"] = (df["latitude"]  / resolution).round() * resolution
    df["grid_lon"] = (df["longitude"] / resolution).round() * resolution

    grid = (
        df.groupby(["grid_lat", "grid_lon"])
        .agg(
            crime_count=("crime_type",   "count"),
            avg_severity=("severity_norm","mean"),
        )
        .reset_index()
    )

    # Density = weighted sum
    max_count = grid["crime_count"].max()
    grid["density"] = (
        0.6 * grid["crime_count"] / max_count +
        0.4 * grid["avg_severity"]
    ).round(4)

    heatmap_data = grid[["grid_lat", "grid_lon", "density"]].rename(
        columns={"grid_lat": "lat", "grid_lon": "lon"}
    ).to_dict(orient="records")

    print(f"✅ Computed density for {len(heatmap_data)} grid cells")
    return heatmap_data


# ─── Time-based risk factor ────────────────────────────────────────────────────
def time_risk_factor(hour: int) -> float:
    """Returns risk multiplier 0.0–1.0 based on hour of day."""
    risk_map = {
        0: 0.95, 1: 0.98, 2: 0.99, 3: 1.0,  4: 0.90, 5: 0.75,
        6: 0.55, 7: 0.40, 8: 0.35, 9: 0.30, 10: 0.28, 11: 0.25,
        12: 0.30, 13: 0.28, 14: 0.25, 15: 0.28, 16: 0.35, 17: 0.45,
        18: 0.60, 19: 0.72, 20: 0.82, 21: 0.88, 22: 0.92, 23: 0.94,
    }
    return risk_map.get(hour % 24, 0.5)


if __name__ == "__main__":
    df = load_and_clean()
    df, clusters = cluster_hotspots(df)
    grid = compute_density_grid(df)
    print(clusters.head())
