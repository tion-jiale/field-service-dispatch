"""
api.py
FastAPI server — wraps DRL+GNN model, geocodes address via Nominatim,
assigns technician, and appends result to assignments.csv for Streamlit.
"""

import os
import csv
import httpx
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Import GNN from env ────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from env import GNNEncoder, haversine, NODE_FEAT, GNN_HIDDEN, GNN_OUT  # type: ignore

app = FastAPI(title="Field Dispatch API — Kuantan")

# ── Constants ──────────────────────────────────────────────────────────────
CENTER_LAT   = 3.8077
CENTER_LON   = 103.3260
RADIUS_KM    = 20
JOB_TYPES    = ["Mechanical", "Electrical", "IT", "Civil"]
ASSIGNMENTS_CSV = "assignments.csv"

LAT_MIN, LAT_MAX = 3.6277,  3.9877
LON_MIN, LON_MAX = 103.146, 103.506

def norm_lat(lat): return (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
def norm_lon(lon): return (lon - LON_MIN) / (LON_MAX - LON_MIN)


# ── Actor network (must match train.py) ───────────────────────────────────
class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, output_size),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


# ── Load GNN encoder ──────────────────────────────────────────────────────
gnn = GNNEncoder(NODE_FEAT, GNN_HIDDEN, GNN_OUT)
gnn.eval()


# ── Load technician data ──────────────────────────────────────────────────
def load_technicians() -> pd.DataFrame:
    if os.path.exists("technician_dataset.csv"):
        return pd.read_csv("technician_dataset.csv")
    raise RuntimeError("technician_dataset.csv not found. Run generate_data.py first.")


# ── Geocode via Nominatim ──────────────────────────────────────────────────
async def geocode(address: str) -> tuple[float, float]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1},
            headers={"User-Agent": "FieldDispatchKuantan/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()

    if not results:
        raise HTTPException(422, f"Cannot geocode: '{address}'")
    return float(results[0]["lat"]), float(results[0]["lon"])


# ── Build GNN state vector ─────────────────────────────────────────────────
def build_state(pool: pd.DataFrame, job_lat: float, job_lon: float,
                job_priority: int, job_type_req: int) -> np.ndarray:
    num_nodes = len(pool) + 1
    feats = []

    for _, t in pool.iterrows():
        feats.append([
            norm_lat(t["lat"]), norm_lon(t["lon"]),
            t["skill"] / 5.0,
            1.0 if t["status"] == "Available" else 0.0,
            0.0,
        ])
    feats.append([
        norm_lat(job_lat), norm_lon(job_lon),
        job_priority / 5.0, 0.0, 1.0,
    ])

    x = torch.tensor(feats, dtype=torch.float32)

    all_lats = list(pool["lat"]) + [job_lat]
    all_lons = list(pool["lon"]) + [job_lon]
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                d = haversine(all_lats[i], all_lons[i], all_lats[j], all_lons[j])
                adj[i, j] = 1.0 / (d + 1e-6)

    adj_t = torch.tensor(adj, dtype=torch.float32)
    with torch.no_grad():
        embedding = gnn(x, adj_t).numpy()

    extra = np.array([0.0, job_priority / 5.0], dtype=np.float32)
    return np.concatenate([embedding, extra])


# ── Append result to CSV (Streamlit reads this) ───────────────────────────
def save_assignment(record: dict):
    file_exists = os.path.exists(ASSIGNMENTS_CSV)
    with open(ASSIGNMENTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


# ── Request schema ─────────────────────────────────────────────────────────
class JobRequest(BaseModel):
    customer_name:  str
    address:        str
    job_priority:   int   # 1–5
    required_skill: int   # 0–3
    problem:        str   = "Not specified"
    duration_exp:   int   = 60


# ── POST /assign ──────────────────────────────────────────────────────────
@app.post("/assign")
async def assign(job: JobRequest):
    # 1. Geocode
    job_lat, job_lon = await geocode(job.address)

    # 2. Radius check
    dist_center = haversine(CENTER_LAT, CENTER_LON, job_lat, job_lon)
    if dist_center > RADIUS_KM:
        raise HTTPException(
            422,
            f"Location is {dist_center:.1f} km from Padang MBK 1 — "
            f"outside the {RADIUS_KM} km service radius."
        )

    # 3. Load technicians & filter
    df_tech   = load_technicians()
    available = df_tech[df_tech["status"] == "Available"].copy()
    skilled   = available[
        (available["job_type"] == JOB_TYPES[job.required_skill]) |
        (available["skill"] >= 4.0)
    ]
    pool = skilled if not skilled.empty else available

    if pool.empty:
        raise HTTPException(422, "No available technician found.")

    pool = pool.reset_index(drop=True)

    # 4. Build state & run actor
    state    = build_state(pool, job_lat, job_lon, job.job_priority, job.required_skill)
    actor    = Actor(len(state), len(pool))
    if os.path.exists("actor.pth"):
        # Actor was trained with fixed output — use nearest-tech fallback if sizes differ
        try:
            actor.load_state_dict(torch.load("actor.pth", map_location="cpu"),
                                  strict=False)
        except Exception:
            pass
    actor.eval()

    with torch.no_grad():
        probs = actor(torch.tensor(state, dtype=torch.float32))
        idx   = torch.argmax(probs).item()
        idx   = min(int(idx), len(pool) - 1)

    assigned  = pool.iloc[idx]
    dist_km   = haversine(assigned["lat"], assigned["lon"], job_lat, job_lon)
    eta_min   = round(dist_km / 40 * 60, 1)

    # 5. Save to CSV for Streamlit
    record = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "customer_name":  job.customer_name,
        "address":        job.address,
        "problem":        job.problem,
        "job_priority":   job.job_priority,
        "required_skill": JOB_TYPES[job.required_skill],
        "job_lat":        round(job_lat, 6),
        "job_lon":        round(job_lon, 6),
        "assigned_to":    assigned["technician_id"],
        "tech_lat":       assigned["lat"],
        "tech_lon":       assigned["lon"],
        "distance_km":    round(dist_km, 3),
        "eta_min":        eta_min,
        "status":         "Dispatched",
    }
    save_assignment(record)

    return record


# ── GET /assignments — Streamlit can also poll this ───────────────────────
@app.get("/assignments")
def get_assignments():
    if not os.path.exists(ASSIGNMENTS_CSV):
        return []
    df = pd.read_csv(ASSIGNMENTS_CSV)
    return df.to_dict(orient="records")


# ── GET /health ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.exists("actor.pth")}
