"""
Namma Safe BLR — Safety Scoring Engine
Assigns a composite risk score to any geographic point.

Safety Score formula (0 = safest, 1 = most dangerous):
  Score = 0.40 × crime_density
        + 0.20 × (1 − lighting_score)
        + 0.20 × (1 − cctv_score)
        + 0.10 × (1 − crowd_density)
        + 0.10 × time_risk
"""

import numpy as np
from typing import Optional
from data_processing import time_risk_factor

# ─── Weight constants ──────────────────────────────────────────────────────────
W_CRIME    = 0.40
W_LIGHT    = 0.20
W_CCTV     = 0.20
W_CROWD    = 0.10
W_TIME     = 0.10


# ─── Precomputed density grid (loaded at startup) ─────────────────────────────
_density_grid: list = []      # list of {lat, lon, density}

def load_density_grid(grid: list):
    """Called by app.py after computing density from crime data."""
    global _density_grid
    _density_grid = grid
    print(f"✅ Safety engine loaded {len(grid)} density cells")


def _nearest_density(lat: float, lon: float) -> float:
    """Look up crime density for the closest grid cell."""
    if not _density_grid:
        return 0.5          # fallback if grid not loaded

    best_d = float("inf")
    best_v = 0.5
    for cell in _density_grid:
        d = (cell["lat"] - lat) ** 2 + (cell["lon"] - lon) ** 2
        if d < best_d:
            best_d = d
            best_v = cell["density"]
    return best_v


# ─── Point-level safety score ─────────────────────────────────────────────────
def compute_safety_score(
    lat: float,
    lon: float,
    hour: int,
    lighting_score: Optional[float] = None,
    cctv_score:     Optional[float] = None,
    crowd_density:  Optional[float] = None,
) -> dict:
    """
    Returns a dict with the composite risk score and component breakdown.

    Args:
        lat, lon        – WGS-84 coordinates
        hour            – Hour of day (0-23)
        lighting_score  – 0 (no lights) to 1 (well lit);  auto-estimated if None
        cctv_score      – 0 (no cameras) to 1 (full coverage); auto-estimated if None
        crowd_density   – 0 (isolated) to 1 (crowded);    auto-estimated if None

    Returns:
        {score, crime_density, lighting, cctv, crowd, time_risk, risk_level}
    """
    crime_density = _nearest_density(lat, lon)
    time_risk     = time_risk_factor(hour)

    # Estimate environmental factors if not provided
    # (heuristics: central Bangalore has more light/CCTV)
    if lighting_score is None:
        # More central areas (near 12.97, 77.59) tend to be better lit
        dist_center = np.sqrt((lat - 12.9716)**2 + (lon - 77.5946)**2)
        lighting_score = max(0.1, 1.0 - dist_center * 8)

    if cctv_score is None:
        cctv_score = max(0.05, lighting_score * 0.7 + np.random.uniform(-0.1, 0.1))

    if crowd_density is None:
        # Night-time = lower crowd
        crowd_density = max(0.1, 0.8 - time_risk * 0.5)

    # Clamp all to [0,1]
    lighting_score = float(np.clip(lighting_score, 0.0, 1.0))
    cctv_score     = float(np.clip(cctv_score,     0.0, 1.0))
    crowd_density  = float(np.clip(crowd_density,  0.0, 1.0))

    score = (
        W_CRIME * crime_density +
        W_LIGHT * (1 - lighting_score) +
        W_CCTV  * (1 - cctv_score) +
        W_CROWD * (1 - crowd_density) +
        W_TIME  * time_risk
    )
    score = float(np.clip(score, 0.0, 1.0))

    # Categorical label
    if score < 0.30:
        risk_level = "LOW"
    elif score < 0.55:
        risk_level = "MEDIUM"
    elif score < 0.75:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return {
        "score":          round(score, 4),
        "score_100":      round(score * 100, 1),
        "crime_density":  round(crime_density, 4),
        "lighting":       round(lighting_score, 4),
        "cctv":           round(cctv_score, 4),
        "crowd":          round(crowd_density, 4),
        "time_risk":      round(time_risk, 4),
        "risk_level":     risk_level,
    }


# ─── Segment-level safety score ───────────────────────────────────────────────
def segment_safety_score(
    p1: tuple,
    p2: tuple,
    hour: int,
) -> float:
    """
    Returns averaged risk score for a road segment (midpoint approximation).
    p1, p2 = (lat, lon) tuples.
    """
    mid_lat = (p1[0] + p2[0]) / 2
    mid_lon = (p1[1] + p2[1]) / 2
    result  = compute_safety_score(mid_lat, mid_lon, hour)
    return result["score"]
