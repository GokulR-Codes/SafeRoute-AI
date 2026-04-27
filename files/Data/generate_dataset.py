"""
Namma Safe BLR — Synthetic Crime Dataset Generator
Generates realistic crime data clustered around known Bangalore hotspot areas.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ─── Realistic Bangalore crime hotspot clusters ───────────────────────────────
HOTSPOT_CLUSTERS = [
    {"name": "Majestic / KSR",       "lat": 12.9768, "lon": 77.5713, "weight": 0.12, "radius": 0.018},
    {"name": "BTM Layout",           "lat": 12.9165, "lon": 77.6101, "weight": 0.09, "radius": 0.015},
    {"name": "Shivajinagar",         "lat": 12.9840, "lon": 77.5975, "weight": 0.08, "radius": 0.014},
    {"name": "Chickpet",             "lat": 12.9675, "lon": 77.5773, "weight": 0.08, "radius": 0.013},
    {"name": "Domlur",               "lat": 12.9609, "lon": 77.6387, "weight": 0.07, "radius": 0.012},
    {"name": "Koramangala",          "lat": 12.9352, "lon": 77.6245, "weight": 0.07, "radius": 0.016},
    {"name": "Marathahalli",         "lat": 12.9591, "lon": 77.6971, "weight": 0.07, "radius": 0.017},
    {"name": "Bannerghatta Road",    "lat": 12.8993, "lon": 77.5975, "weight": 0.06, "radius": 0.015},
    {"name": "Indiranagar",          "lat": 12.9716, "lon": 77.6412, "weight": 0.06, "radius": 0.014},
    {"name": "Hebbal",               "lat": 13.0358, "lon": 77.5970, "weight": 0.05, "radius": 0.015},
    {"name": "Whitefield",           "lat": 12.9698, "lon": 77.7499, "weight": 0.05, "radius": 0.018},
    {"name": "Electronic City",      "lat": 12.8452, "lon": 77.6602, "weight": 0.04, "radius": 0.016},
    {"name": "Yelahanka",            "lat": 13.1006, "lon": 77.5964, "weight": 0.04, "radius": 0.015},
    {"name": "Rajajinagar",          "lat": 12.9940, "lon": 77.5524, "weight": 0.04, "radius": 0.013},
    {"name": "Jayanagar",            "lat": 12.9258, "lon": 77.5838, "weight": 0.04, "radius": 0.014},
    {"name": "Random Spread",        "lat": 12.9716, "lon": 77.5946, "weight": 0.04, "radius": 0.060},
]

CRIME_TYPES = {
    "Theft":              {"severity": (2, 6),  "time_peak": [19, 20, 21, 22, 23, 0, 1]},
    "Chain Snatching":    {"severity": (4, 7),  "time_peak": [7, 8, 17, 18, 19, 20, 21]},
    "Robbery":            {"severity": (6, 9),  "time_peak": [21, 22, 23, 0, 1, 2]},
    "Eve Teasing":        {"severity": (3, 7),  "time_peak": [18, 19, 20, 21, 22, 23]},
    "Vehicle Theft":      {"severity": (3, 6),  "time_peak": [21, 22, 23, 0, 1]},
    "Assault":            {"severity": (5, 9),  "time_peak": [20, 21, 22, 23, 0, 1, 2]},
    "Pickpocketing":      {"severity": (2, 5),  "time_peak": [8, 9, 10, 17, 18, 19]},
    "Cybercrime":         {"severity": (3, 7),  "time_peak": list(range(24))},
    "Drug Offense":       {"severity": (5, 8),  "time_peak": [20, 21, 22, 23, 0, 1, 2, 3]},
    "Vandalism":          {"severity": (2, 4),  "time_peak": [22, 23, 0, 1, 2, 3]},
}

def generate_records(n=1500):
    records = []
    start_date = datetime(2023, 1, 1)
    end_date   = datetime(2024, 12, 31)

    cluster_names  = [c["name"]   for c in HOTSPOT_CLUSTERS]
    cluster_weights = [c["weight"] for c in HOTSPOT_CLUSTERS]

    for _ in range(n):
        # Pick cluster
        idx = np.random.choice(len(HOTSPOT_CLUSTERS), p=cluster_weights)
        cluster = HOTSPOT_CLUSTERS[idx]

        lat = cluster["lat"] + np.random.normal(0, cluster["radius"])
        lon = cluster["lon"] + np.random.normal(0, cluster["radius"])

        # Pick crime type
        crime_type = random.choice(list(CRIME_TYPES.keys()))
        meta = CRIME_TYPES[crime_type]

        # Bias time toward crime peaks
        if random.random() < 0.65:
            hour = random.choice(meta["time_peak"])
        else:
            hour = random.randint(0, 23)
        minute = random.randint(0, 59)

        severity = round(random.uniform(*meta["severity"]), 1)

        # Random date
        days_offset = random.randint(0, (end_date - start_date).days)
        date = start_date + timedelta(days=days_offset)

        # Derived features
        lighting_score  = round(random.uniform(0.1, 1.0), 2)
        cctv_score      = round(random.uniform(0.0, 1.0), 2)
        crowd_density   = round(random.uniform(0.1, 1.0), 2)
        police_proximity = round(random.uniform(0.1, 1.0), 2)

        is_night = 1 if (hour >= 20 or hour < 6) else 0

        records.append({
            "latitude":          round(lat, 6),
            "longitude":         round(lon, 6),
            "crime_type":        crime_type,
            "crime_severity":    severity,
            "date":              date.strftime("%Y-%m-%d"),
            "time":              f"{hour:02d}:{minute:02d}",
            "hour":              hour,
            "area":              cluster["name"].split("/")[0].strip(),
            "lighting_score":    lighting_score,
            "cctv_score":        cctv_score,
            "crowd_density":     crowd_density,
            "police_proximity":  police_proximity,
            "is_night":          is_night,
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = generate_records(1500)
    df.to_csv("bangalore_crime_dataset.csv", index=False)
    print(f"✅ Dataset generated: {len(df)} records")
    print(df.head())
    print("\nCrime type distribution:")
    print(df["crime_type"].value_counts())
