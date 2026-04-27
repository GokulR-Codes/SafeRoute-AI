"""
Namma Safe BLR — Analytics Dashboard
PowerBI-style crime analytics exported as HTML dashboard panels.
Run: python crime_dashboard.py
"""

import os, sys, json
import pandas as pd
import numpy  as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))
from data_processing import load_and_clean, time_risk_factor

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/bangalore_crime_dataset.csv")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "../analytics")

# ─── Dark dashboard theme ─────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
ACCENT   = "#f78166"
ACCENT2  = "#79c0ff"
ACCENT3  = "#56d364"
ACCENT4  = "#d2a8ff"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
})

CRIME_COLORS = {
    "Theft":           "#f78166",
    "Chain Snatching": "#ffa657",
    "Robbery":         "#ff7b72",
    "Eve Teasing":     "#d2a8ff",
    "Vehicle Theft":   "#79c0ff",
    "Assault":         "#f85149",
    "Pickpocketing":   "#56d364",
    "Cybercrime":      "#58a6ff",
    "Drug Offense":    "#bc8cff",
    "Vandalism":       "#8b949e",
}


def make_dashboard(df: pd.DataFrame, out_path: str):
    fig = plt.figure(figsize=(20, 14), facecolor=DARK_BG)
    fig.suptitle(
        "  🛡️  NAMMA SAFE BLR — Crime Intelligence Dashboard",
        fontsize=22, fontweight="bold", color=TEXT, y=0.98,
        fontfamily="monospace",
    )

    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38,
                  top=0.92, bottom=0.06, left=0.06, right=0.97)

    # ─ KPI tiles row (top) ──────────────────────────────────────────────────
    kpi_data = [
        ("TOTAL CRIMES",  len(df),                                     ACCENT,  "▲"),
        ("NIGHT CRIMES",  int(df["is_night"].sum()),                    "#f85149","🌙"),
        ("HIGH RISK AREAS",int((df["crime_severity"] >= 7).sum()),      ACCENT4, "⚠"),
        ("AREAS COVERED",  df["area"].nunique(),                        ACCENT3, "📍"),
    ]
    for i, (title, val, color, icon) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(PANEL_BG)
        # Card border
        rect = mpatches.FancyBboxPatch((0.05, 0.05), 0.90, 0.90,
            boxstyle="round,pad=0.02", linewidth=2,
            edgecolor=color, facecolor=PANEL_BG)
        ax.add_patch(rect)
        ax.text(0.5, 0.72, icon,   fontsize=24, ha="center", va="center", color=color)
        ax.text(0.5, 0.48, str(val), fontsize=26, fontweight="bold",
                ha="center", va="center", color=color)
        ax.text(0.5, 0.22, title,  fontsize=8,  ha="center", va="center",
                color=MUTED, fontweight="bold")

    # ─ Crime frequency by hour ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, :2])
    hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)
    bars = ax1.bar(hourly.index, hourly.values, color=ACCENT2, alpha=0.85, width=0.75)
    # Colour night hours red
    for bar, h in zip(bars, hourly.index):
        if h >= 20 or h < 6:
            bar.set_color(ACCENT)
            bar.set_alpha(0.9)
    ax1.set_title("Crime Frequency by Hour of Day", fontweight="bold", pad=10)
    ax1.set_xlabel("Hour (24h)")
    ax1.set_ylabel("Incident Count")
    ax1.set_xticks(range(24))
    ax1.axvspan(20, 24, alpha=0.08, color=ACCENT, label="Night Zone")
    ax1.axvspan(0,  6,  alpha=0.08, color=ACCENT)
    ax1.legend(fontsize=8, facecolor=PANEL_BG)
    ax1.grid(axis="y")

    # ─ Crime type distribution (horizontal bar) ──────────────────────────────
    ax2 = fig.add_subplot(gs[1, 2:])
    type_counts = df["crime_type"].value_counts()
    colors = [CRIME_COLORS.get(t, ACCENT2) for t in type_counts.index]
    ax2.barh(type_counts.index, type_counts.values, color=colors, alpha=0.9)
    ax2.set_title("Crime Type Distribution", fontweight="bold", pad=10)
    ax2.set_xlabel("Incident Count")
    ax2.grid(axis="x")
    for i, (idx, val) in enumerate(type_counts.items()):
        ax2.text(val + 2, i, str(val), va="center", fontsize=8, color=MUTED)

    # ─ Top 8 areas (area heatbar) ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :2])
    area_counts = df.groupby("area").size().sort_values(ascending=False).head(8)
    max_val = area_counts.max()
    bar_colors = plt.cm.RdYlGn_r(area_counts.values / max_val)
    ax3.barh(area_counts.index[::-1], area_counts.values[::-1],
             color=bar_colors[::-1], alpha=0.9)
    ax3.set_title("High-Risk Areas — Crime Count", fontweight="bold", pad=10)
    ax3.set_xlabel("Incidents")
    ax3.grid(axis="x")

    # ─ Night vs Day pie ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 2])
    night = int(df["is_night"].sum())
    day   = len(df) - night
    wedge_props = {"linewidth": 2, "edgecolor": DARK_BG}
    ax4.pie(
        [night, day],
        labels     = [f"Night\n{night}", f"Day\n{day}"],
        colors     = [ACCENT, ACCENT3],
        autopct    = "%1.1f%%",
        startangle = 90,
        wedgeprops = wedge_props,
        textprops  = {"color": TEXT, "fontsize": 9},
    )
    ax4.set_title("Night vs Day Crimes", fontweight="bold", pad=10)

    # ─ Severity distribution ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 3])
    sev  = df["crime_severity"].values
    n, bins, patches = ax5.hist(sev, bins=15, color=ACCENT4, alpha=0.85, edgecolor=DARK_BG)
    # Gradient colour by severity
    for p, b in zip(patches, bins):
        norm_val = b / 10
        p.set_facecolor(plt.cm.RdYlGn_r(norm_val))
    ax5.set_title("Crime Severity Distribution", fontweight="bold", pad=10)
    ax5.set_xlabel("Severity Score (1–10)")
    ax5.set_ylabel("Count")
    ax5.grid(axis="y")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"✅ Dashboard saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df  = load_and_clean(DATA_PATH)
    out = os.path.join(OUT_DIR, "crime_dashboard.png")
    make_dashboard(df, out)
