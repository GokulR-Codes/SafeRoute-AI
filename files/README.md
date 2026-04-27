# 🛡️ SAFE ROUTE - AI
## AI-Powered Safe Navigation System for Night Travel

> **IEEE/Hackathon Grade Project** | Women's Safety | Urban Security | AI Navigation

---

## 🎯 Project Overview

SafeRoute AI is an AI-assisted navigation system that prioritizes **safety over speed** by analyzing:
- Crime density and hotspot patterns (DBSCAN clustering)
- Street lighting availability
- CCTV coverage estimation
- Time-based risk prediction (crime peaks by hour)
- Police station proximity

### What makes it different from Google Maps?
| Feature | Google Maps | Namma Safe BLR |
|---|---|---|
| Route optimization | Distance / Time | **Safety Score** |
| Crime awareness | ❌ | ✅ DBSCAN hotspot clustering |
| Time-risk factor | ❌ | ✅ Hourly risk model |
| Night mode | Basic | ✅ Auto-activated, risk-adjusted |
| ML risk prediction | ❌ | ✅ 96.7% accuracy |

---

## 🏗️ Architecture

```
SafeRoute AI/
├── backend/
│   ├── app.py              # FastAPI REST API server
│   ├── data_processing.py  # Crime data loading, DBSCAN clustering, density grids
│   ├── safety_score.py     # Composite safety scoring engine
│   ├── risk_model.py       # ML training (RF / GBM / LR) + inference
│   └── route_engine.py     # Modified A* / Dijkstra with risk cost
├── frontend/
│   └── index.html          # Self-contained SPA (Leaflet.js + embedded data)
├── data/
│   ├── bangalore_crime_dataset.csv   # 1500-record synthetic crime dataset
│   └── generate_dataset.py           # Dataset generator (re-run to regenerate)
├── models/
│   └── risk_model.pkl      # Trained sklearn pipeline (auto-generated)
└── analytics/
    ├── crime_dashboard.py  # PowerBI-style matplotlib dashboard
    └── crime_dashboard.png # Generated dashboard image
```

---

## ⚙️ Setup Instructions

### Prerequisites
```bash
Python >= 3.9
pip
```

### 1. Install dependencies
```bash
pip install fastapi uvicorn pandas numpy scikit-learn matplotlib geopandas shapely
```

### 2. Generate crime dataset (already included)
```bash
cd data/
python generate_dataset.py
```

### 3. Train / verify ML model
```bash
cd backend/
python risk_model.py
# Expected output: Best model accuracy ~96-97%
```

### 4. Generate analytics dashboard
```bash
cd analytics/
python crime_dashboard.py
# Outputs: analytics/crime_dashboard.png
```

### 5. Launch the API server
```bash
cd backend/
uvicorn app:app --reload --port 8000
# API docs available at: http://localhost:8000/docs
```

### 6. Open the frontend
Simply open `frontend/index.html` in any modern browser.  
No server needed — all data is embedded!

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/find-safe-route` | A* safe route with risk scoring |
| POST | `/predict-risk` | ML prediction for any location |
| GET | `/get-crime-heatmap` | Crime density grid data |
| GET | `/get-crime-points` | Raw crime records for markers |
| POST | `/report-incident` | Crowd-sourced incident submission |
| GET | `/get-incidents` | All crowd reports |
| GET | `/analytics/summary` | Crime analytics summary |

### Example: Find Safe Route
```bash
curl -X POST http://localhost:8000/find-safe-route \
  -H "Content-Type: application/json" \
  -d '{
    "src_lat": 12.9716, "src_lon": 77.5946,
    "dst_lat": 12.9352, "dst_lon": 77.6245,
    "hour": 22,
    "algorithm": "astar"
  }'
```

### Example: Predict Risk
```bash
curl -X POST http://localhost:8000/predict-risk \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 12.9768, "longitude": 77.5713,
    "hour": 22,
    "lighting_score": 0.3,
    "cctv_score": 0.2
  }'
```

---

## 🧠 ML Model

**Training Data:** 1,500 synthetic Bangalore crime records  
**Features:** crime_density, time_risk, lighting_score, cctv_score, crowd_density, police_proximity, is_night, hour  
**Models Evaluated:**
- Random Forest (CV: ~87.9%)
- Gradient Boosting (CV: ~89.1%)
- **Logistic Regression (CV: ~97.1%) ← Best**

**Output Classes:** LOW / MEDIUM / HIGH / CRITICAL

---

## 🗺️ Safety Score Formula

```
Safety Score = 0.40 × crime_density
             + 0.20 × (1 − lighting_score)
             + 0.20 × (1 − cctv_score)
             + 0.10 × (1 − crowd_density)
             + 0.10 × time_risk_factor(hour)

Score range: 0.0 (safest) → 1.0 (most dangerous)
```

**Risk Levels:**
- 🟢 LOW: score < 0.30
- 🟡 MEDIUM: 0.30 – 0.55
- 🔴 HIGH: 0.55 – 0.75
- ⛔ CRITICAL: score > 0.75

---

## 🛣️ Route Algorithm

Modified **A*** where edge cost = blended safety + distance:
```
edge_cost = 0.75 × risk_score + 0.25 × (distance / 20km)
heuristic = geographic_distance × 0.25   (admissible)
```

The result is a route that strongly prefers low-risk roads over shorter but dangerous alternatives.

---

## 🌙 Features

| Feature | Status |
|---|---|
| Crime heatmap (Leaflet) | ✅ |
| Safe route A* algorithm | ✅ |
| ML risk prediction | ✅ |
| Police station overlay | ✅ |
| Night mode auto-detection | ✅ |
| SOS emergency button | ✅ |
| Crowd incident reporting | ✅ |
| Analytics dashboard | ✅ |
| REST API | ✅ |
| Dark map theme | ✅ |

---

## 📊 Dataset

Synthetic 1,500-record Bangalore crime dataset with realistic spatial clustering.

**Crime types:** Theft, Chain Snatching, Robbery, Eve Teasing, Vehicle Theft, Assault, Pickpocketing, Cybercrime, Drug Offense, Vandalism

**Hotspot clusters:** Majestic, BTM Layout, Shivajinagar, Chickpet, Domlur, Koramangala, Marathahalli, Bannerghatta, Indiranagar, Hebbal, Whitefield, Electronic City, Yelahanka, Rajajinagar, Jayanagar

---

## 🏆 IEEE/Hackathon Demo Notes

For live demo, start with:
1. Show crime heatmap → identify red danger zones
2. Enable crime incident markers → see clustering
3. Select MG Road → BTM Layout route at 22:00
4. Click "Find Safest Route" → watch A* avoid red zones
5. Compare route panel: risk score, distance, waypoints
6. Show analytics dashboard image (crime_dashboard.png)
7. Demo SOS button + incident reporting

**Tech stack highlights for judges:**
- Geospatial AI: DBSCAN clustering, grid density mapping
- ML Pipeline: Scikit-learn with cross-validation model selection
- Graph Algorithms: A* with custom safety cost function
- REST API: FastAPI with Pydantic validation
- Frontend: Leaflet.js + custom dark UI

---


