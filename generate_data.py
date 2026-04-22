"""
generate_data.py
Generates all 4 simulated datasets for the Field Service Dispatch system.
Scope: Petrol stations within 20km of Padang MBK 1, Kuantan, Pahang.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# ── Reference point: Padang MBK 1, Kuantan ────────────────────────────────
CENTER_LAT = 3.8077
CENTER_LON = 103.3260
RADIUS_KM  = 20

# ── Haversine distance ─────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ── Generate random point within radius ──────────────────────────────────
def random_point_within_radius(center_lat, center_lon, radius_km):
    while True:
        # Random offset in degrees (~111km per degree)
        dlat = np.random.uniform(-radius_km/111, radius_km/111)
        dlon = np.random.uniform(-radius_km/111, radius_km/111)
        lat  = center_lat + dlat
        lon  = center_lon + dlon
        if haversine(center_lat, center_lon, lat, lon) <= radius_km:
            return round(lat, 6), round(lon, 6)

# ═══════════════════════════════════════════════════════════════════════════
# 1. TECHNICIAN DATASET
# ═══════════════════════════════════════════════════════════════════════════
NUM_TECHS = 10
job_types  = ["Mechanical", "Electrical", "IT", "Civil"]
shifts     = ["Morning", "Evening"]
statuses   = ["Available", "Working", "Off Shift"]

technicians = []
for i in range(1, NUM_TECHS + 1):
    lat, lon = random_point_within_radius(CENTER_LAT, CENTER_LON, RADIUS_KM)
    technicians.append({
        "technician_id":    f"TECH{i:03d}",
        "technician_shift": random.choice(shifts),
        "skill":            round(random.uniform(1, 5), 1),   # 1.0–5.0
        "status":           random.choice(statuses),
        "job_type":         random.choice(job_types),
        "experience":       random.randint(1, 20),             # years
        "lat":              lat,
        "lon":              lon,
    })

df_tech = pd.DataFrame(technicians)
df_tech.to_csv("technician_dataset.csv", index=False)
print(f"✅ technician_dataset.csv — {len(df_tech)} rows")


# ═══════════════════════════════════════════════════════════════════════════
# 2. WORKLOAD DATASET
# ═══════════════════════════════════════════════════════════════════════════
workloads = []
for tech in technicians:
    workloads.append({
        "tech_id":      tech["technician_id"],
        "no_of_jobs":   random.randint(0, 15),
        "service_time": round(random.uniform(30, 240), 1),   # minutes
        "travel_time":  round(random.uniform(5, 90), 1),     # minutes
    })

df_workload = pd.DataFrame(workloads)
df_workload.to_csv("workload_dataset.csv", index=False)
print(f"✅ workload_dataset.csv — {len(df_workload)} rows")


# ═══════════════════════════════════════════════════════════════════════════
# 3. JOB DATASET
# ═══════════════════════════════════════════════════════════════════════════
NUM_JOBS = 50
problems = [
    "Fuel pump failure",
    "POS terminal not working",
    "CCTV system down",
    "Air compressor breakdown",
    "Generator fault",
    "Electrical wiring issue",
    "Canopy lighting failure",
    "Underground tank sensor error",
    "Fire suppression system fault",
    "Car wash machine breakdown",
]
job_statuses = ["Submitted", "Ongoing", "Completed"]

base_time = datetime(2025, 1, 1, 8, 0, 0)
jobs = []
for i in range(1, NUM_JOBS + 1):
    lat, lon = random_point_within_radius(CENTER_LAT, CENTER_LON, RADIUS_KM)
    ts = base_time + timedelta(hours=random.randint(0, 720))
    jobs.append({
        "job_id":       f"JOB{i:04d}",
        "priority":     random.randint(1, 5),
        "problems":     random.choice(problems),
        "timestamp":    ts.strftime("%Y-%m-%d %H:%M:%S"),
        "lat":          lat,
        "lon":          lon,
        "duration_exp": random.randint(30, 180),   # minutes
        "status":       random.choice(job_statuses),
    })

df_job = pd.DataFrame(jobs)
df_job.to_csv("job_dataset.csv", index=False)
print(f"✅ job_dataset.csv — {len(df_job)} rows")


# ═══════════════════════════════════════════════════════════════════════════
# 4. SUPERVISION DATASET
# ═══════════════════════════════════════════════════════════════════════════
supervision = []
completed_jobs = df_job[df_job["status"].isin(["Ongoing", "Completed"])].copy()

for _, job in completed_jobs.iterrows():
    tech = random.choice(technicians)
    req_ts   = datetime.strptime(job["timestamp"], "%Y-%m-%d %H:%M:%S")
    dist_km  = haversine(tech["lat"], tech["lon"], job["lat"], job["lon"])
    travel   = round(dist_km / 40 * 60, 1)     # assume 40 km/h avg speed → minutes
    arrival  = req_ts + timedelta(minutes=travel)
    duration = round(random.uniform(30, job["duration_exp"] * 1.2), 1)
    complete = arrival + timedelta(minutes=duration)
    cust_wait = round((arrival - req_ts).total_seconds() / 60, 1)

    supervision.append({
        "tech_id":              tech["technician_id"],
        "job_id":               job["job_id"],
        "req_timestamp":        job["timestamp"],
        "dist":                 round(dist_km, 3),
        "duration":             duration,
        "cust_wait_time":       cust_wait,
        "exp_time_remaining":   max(0, round(job["duration_exp"] - duration, 1)),
        "arrival_time_only":    arrival.strftime("%H:%M:%S"),
        "completion_time_only": complete.strftime("%H:%M:%S"),
    })

df_supervision = pd.DataFrame(supervision)
df_supervision.to_csv("supervision_dataset.csv", index=False)
print(f"✅ supervision_dataset.csv — {len(df_supervision)} rows")

print("\nAll datasets generated successfully.")
