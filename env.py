"""
env.py — Fixed FieldDispatchEnv
Key fixes:
1. Reward rescaled — balanced distance vs priority signals
2. Episode ends immediately when all jobs done (no drag)
3. Skill match guaranteed — at least one tech can do each job
4. GNN embeddings normalized — reduces noise for actor
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Haversine distance (km) ────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ── Module-level GNN constants ─────────────────────────────────────────────
NODE_FEAT  = 5
GNN_HIDDEN = 16
GNN_OUT    = 8

# ── Real petrol station fallback ───────────────────────────────────────────
PETROL_STATIONS = [
    {"lat": 3.843602,  "lon": 103.3389947},
    {"lat": 3.9038326, "lon": 103.2468837},
    {"lat": 3.8157123, "lon": 103.3330641},
    {"lat": 3.7968744, "lon": 103.168695 },
    {"lat": 3.9441146, "lon": 103.2420357},
    {"lat": 3.8050617, "lon": 103.3028328},
    {"lat": 3.8295294, "lon": 103.3418005},
    {"lat": 3.8219602, "lon": 103.3265225},
    {"lat": 3.7362248, "lon": 103.3069903},
    {"lat": 3.7561354, "lon": 103.2036068},
    {"lat": 3.7840716, "lon": 103.2880551},
    {"lat": 3.9233963, "lon": 103.366629 },
]
SKILL_MAP  = {"Mechanical": 0, "Electrical": 1, "IT": 2, "Civil": 3}
MAX_DIST_KM = 40.0


# ═══════════════════════════════════════════════════════════════════════════
# GNN
# ═══════════════════════════════════════════════════════════════════════════
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W   = nn.Linear(in_features, out_features, bias=False)
        self.att = nn.Linear(2 * out_features, 1, bias=False)

    def forward(self, x, adj):
        h   = self.W(x)
        N   = h.size(0)
        h_i = h.unsqueeze(1).expand(-1, N, -1)
        h_j = h.unsqueeze(0).expand(N, -1, -1)
        e   = F.leaky_relu(
            self.att(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)
        )
        zero_vec  = -9e15 * torch.ones_like(e)
        attention = F.softmax(torch.where(adj > 0, e, zero_vec), dim=-1)
        return F.elu(
            torch.bmm(attention.unsqueeze(1),
                      h.unsqueeze(0).expand(N, -1, -1)).squeeze(1)
        )


class GNNEncoder(nn.Module):
    def __init__(self, node_feat, hidden, out_feat):
        super().__init__()
        self.layer1 = GraphAttentionLayer(node_feat, hidden)
        self.layer2 = GraphAttentionLayer(hidden, out_feat)

    def forward(self, x, adj):
        h = self.layer2(self.layer1(x, adj), adj)
        # Normalize to reduce noise from random init
        h = (h - h.mean()) / (h.std() + 1e-8)
        return h.flatten()


# ═══════════════════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════════════════
class FieldDispatchEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    LAT_MIN, LAT_MAX = 3.6277,  3.9877
    LON_MIN, LON_MAX = 103.146, 103.506

    def __init__(self, num_techs=5, num_jobs=3, render_mode=None):
        super().__init__()
        self.num_techs   = num_techs
        self.num_jobs    = num_jobs
        self.render_mode = render_mode
        self.max_steps   = num_jobs * 3   # tighter episode limit

        self.num_nodes = num_techs + num_jobs
        self.obs_dim   = self.num_nodes * GNN_OUT + num_jobs * 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_techs * num_jobs)

        self.gnn = GNNEncoder(NODE_FEAT, GNN_HIDDEN, GNN_OUT)
        self.gnn.eval()

        self._load_datasets()

    # ── Load real datasets ─────────────────────────────────────────────────
    def _load_datasets(self):
        self._tech_df = None
        self._job_df  = None
        if os.path.exists("technician_dataset.csv"):
            df = pd.read_csv("technician_dataset.csv")
            self._tech_df = df[df["status"] == "Available"].reset_index(drop=True)
        if os.path.exists("job_dataset.csv"):
            self._job_df = pd.read_csv("job_dataset.csv").reset_index(drop=True)

    # ── Helpers ────────────────────────────────────────────────────────────
    def _norm_lat(self, lat):
        return (lat - self.LAT_MIN) / (self.LAT_MAX - self.LAT_MIN)

    def _norm_lon(self, lon):
        return (lon - self.LON_MIN) / (self.LON_MAX - self.LON_MIN)

    def _rand_station(self):
        s    = PETROL_STATIONS[np.random.randint(len(PETROL_STATIONS))]
        dlat = np.random.uniform(-0.002, 0.002)
        dlon = np.random.uniform(-0.002, 0.002)
        return float(s["lat"] + dlat), float(s["lon"] + dlon)

    # ── Graph construction ─────────────────────────────────────────────────
    def _build_adjacency(self):
        all_lats = np.concatenate([self.tech_lat, self.job_lat])
        all_lons = np.concatenate([self.tech_lon, self.job_lon])
        N   = self.num_nodes
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = haversine(all_lats[i], all_lons[i],
                                  all_lats[j], all_lons[j])
                    adj[i, j] = 1.0 / (d + 1e-6)
        return torch.tensor(adj, dtype=torch.float32)

    def _build_node_features(self):
        feats = []
        for i in range(self.num_techs):
            feats.append([
                self._norm_lat(self.tech_lat[i]),
                self._norm_lon(self.tech_lon[i]),
                self.tech_skill[i] / 5.0,
                float(self.tech_available[i]),
                0.0   # node type: technician
            ])
        for j in range(self.num_jobs):
            feats.append([
                self._norm_lat(self.job_lat[j]),
                self._norm_lon(self.job_lon[j]),
                self.job_priority[j] / 5.0,
                float(self.jobs_done[j]),
                1.0   # node type: job
            ])
        return torch.tensor(feats, dtype=torch.float32)

    def _get_state(self):
        with torch.no_grad():
            embedding = self.gnn(
                self._build_node_features(),
                self._build_adjacency()
            ).numpy()
        extra = np.concatenate([
            self.jobs_done,
            self.job_priority / 5.0
        ]).astype(np.float32)
        return np.concatenate([embedding, extra]).astype(np.float32)

    # ── Reset ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Technicians
        if self._tech_df is not None and len(self._tech_df) >= self.num_techs:
            s = self._tech_df.sample(self.num_techs).reset_index(drop=True)
            self.tech_lat      = s["lat"].values.astype(np.float32)
            self.tech_lon      = s["lon"].values.astype(np.float32)
            self.tech_skill    = s["skill"].values.astype(np.float32)
            self.tech_job_type = np.array([SKILL_MAP.get(j, 0)
                                           for j in s["job_type"]])
        else:
            coords = [self._rand_station() for _ in range(self.num_techs)]
            self.tech_lat      = np.array([c[0] for c in coords], dtype=np.float32)
            self.tech_lon      = np.array([c[1] for c in coords], dtype=np.float32)
            self.tech_skill    = np.random.uniform(1, 5, self.num_techs).astype(np.float32)
            self.tech_job_type = np.random.randint(0, 4, self.num_techs)

        self.tech_available = np.ones(self.num_techs,  dtype=np.float32)
        self.tech_job_count = np.zeros(self.num_techs, dtype=np.float32)

        # Jobs
        if self._job_df is not None and len(self._job_df) >= self.num_jobs:
            s = self._job_df.sample(self.num_jobs).reset_index(drop=True)
            self.job_lat      = s["lat"].values.astype(np.float32)
            self.job_lon      = s["lon"].values.astype(np.float32)
            self.job_priority = s["priority"].values.astype(np.float32)
            self.job_type_req = np.array([SKILL_MAP.get(sk, 0)
                                          for sk in s["required_skill"]])
        else:
            coords = [self._rand_station() for _ in range(self.num_jobs)]
            self.job_lat      = np.array([c[0] for c in coords], dtype=np.float32)
            self.job_lon      = np.array([c[1] for c in coords], dtype=np.float32)
            self.job_priority = np.random.randint(1, 6, self.num_jobs).astype(np.float32)
            self.job_type_req = np.random.randint(0, 4, self.num_jobs)

        self.jobs_done = np.zeros(self.num_jobs, dtype=np.float32)

        # Guarantee at least one tech can do each job
        for j in range(self.num_jobs):
            capable = [i for i in range(self.num_techs)
                       if self.tech_job_type[i] == self.job_type_req[j]
                       or self.tech_skill[i] >= 4.0]
            if not capable:
                self.tech_job_type[np.random.randint(self.num_techs)] = \
                    self.job_type_req[j]

        # Skill matrix
        self.skill_matrix = np.array([
            [1 if (self.tech_job_type[i] == self.job_type_req[j]
                   or self.tech_skill[i] >= 4.0) else 0
             for j in range(self.num_jobs)]
            for i in range(self.num_techs)
        ])

        self.current_step = 0
        return self._get_state(), {}

    # ── Step ───────────────────────────────────────────────────────────────
    def step(self, action):
        self.current_step += 1
        tech_id = min(action // self.num_jobs, self.num_techs - 1)
        job_id  = min(action % self.num_jobs,  self.num_jobs  - 1)

        reward = -0.5   # small step penalty

        if not self.tech_available[tech_id]:
            reward -= 5.0    # busy penalty

        elif self.skill_matrix[tech_id][job_id] == 0:
            reward -= 10.0   # skill mismatch

        elif self.jobs_done[job_id] == 1:
            reward -= 5.0    # duplicate assignment

        else:
            dist_km  = haversine(
                self.tech_lat[tech_id], self.tech_lon[tech_id],
                self.job_lat[job_id],   self.job_lon[job_id]
            )
            priority      = self.job_priority[job_id]
            avg_jobs      = self.tech_job_count.mean()
            fairness      = max(0.0, avg_jobs - self.tech_job_count[tech_id])

            # Balanced reward — all components scaled 0–10
            dist_norm     = dist_km / MAX_DIST_KM           # 0–1
            priority_norm = (priority - 1.0) / 4.0          # 0–1

            reward += (priority_norm * 10.0) \
                    + ((1.0 - dist_norm) * 10.0) \
                    + (fairness * 2.0)

            # Update state
            self.tech_lat[tech_id]       = self.job_lat[job_id]
            self.tech_lon[tech_id]       = self.job_lon[job_id]
            self.tech_available[tech_id] = 0.0
            self.tech_job_count[tech_id] += 1
            self.jobs_done[job_id]       = 1.0

        terminated = bool(np.all(self.jobs_done == 1))
        truncated  = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._get_state(), reward, terminated, truncated, {}

    # ── Render ─────────────────────────────────────────────────────────────
    def render(self):
        print(f"\n── Step {self.current_step} ──────────────────────────")
        for i in range(self.num_techs):
            status = "Available" if self.tech_available[i] else "Busy"
            print(f"  Tech {i:02d} | ({self.tech_lat[i]:.4f}, "
                  f"{self.tech_lon[i]:.4f}) | Skill {self.tech_skill[i]:.1f}"
                  f" | {status}")
        for j in range(self.num_jobs):
            done = "✅" if self.jobs_done[j] else "⏳"
            print(f"  Job  {j:02d} | ({self.job_lat[j]:.4f}, "
                  f"{self.job_lon[j]:.4f}) | P{int(self.job_priority[j])}"
                  f" | {done}")

    def close(self):
        pass
