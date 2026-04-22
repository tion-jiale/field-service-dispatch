"""
env.py
FieldDispatchEnv — DRL + GNN environment for petrol station technician dispatch.
Scope: Kuantan, Pahang. Reference: Padang MBK 1 (3.8077, 103.3260), radius 20km.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
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

# ── Normalize lat/lon to [0, 1] ───────────────────────────────────────────
def norm_lat(lat): return (lat + 90.0)  / 180.0
def norm_lon(lon): return (lon + 180.0) / 360.0

# ═══════════════════════════════════════════════════════════════════════════
# GNN — Graph Attention Layer
# Nodes = technicians + jobs; Edges = distances
# ═══════════════════════════════════════════════════════════════════════════
class GraphAttentionLayer(nn.Module):
    """Single-head graph attention layer (GAT-style)."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W   = nn.Linear(in_features, out_features, bias=False)
        self.att = nn.Linear(2 * out_features, 1, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (N, in_features)  node features
        adj : (N, N)             adjacency / edge weight matrix
        """
        h = self.W(x)                               # (N, out_features)
        N = h.size(0)

        # Pairwise concatenation for attention
        h_i = h.unsqueeze(1).expand(-1, N, -1)      # (N, N, F)
        h_j = h.unsqueeze(0).expand(N, -1, -1)      # (N, N, F)
        e   = F.leaky_relu(self.att(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))

        # Mask non-edges, softmax over neighbours
        zero_vec  = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)     # (N, N)

        return F.elu(torch.bmm(attention.unsqueeze(1), h.unsqueeze(0)
                               .expand(N, -1, -1)).squeeze(1))


class GNNEncoder(nn.Module):
    """
    Two-layer GAT that encodes the dispatch graph.
    Output: flattened embedding of all nodes.
    """
    def __init__(self, node_feat: int, hidden: int, out_feat: int):
        super().__init__()
        self.layer1 = GraphAttentionLayer(node_feat, hidden)
        self.layer2 = GraphAttentionLayer(hidden,   out_feat)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x, adj)
        h = self.layer2(h, adj)
        return h.flatten()   # (N * out_feat,)


# ═══════════════════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════════════════
class FieldDispatchEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    # Spatial bounds (Kuantan ± 20km)
    LAT_MIN, LAT_MAX = 3.6277,  3.9877   # 3.8077 ± ~0.18°
    LON_MIN, LON_MAX = 103.146, 103.506  # 103.326 ± ~0.18°

    # GNN architecture
    NODE_FEAT  = 5    # [norm_lat, norm_lon, skill/priority, availability, type]
    GNN_HIDDEN = 16
    GNN_OUT    = 8

    def __init__(self, num_techs: int = 5, num_jobs: int = 3,
                 render_mode=None):
        super().__init__()
        self.num_techs   = num_techs
        self.num_jobs    = num_jobs
        self.render_mode = render_mode
        self.max_steps   = 50

        # Total nodes in graph: techs + jobs
        self.num_nodes = num_techs + num_jobs
        gnn_embed_size  = self.num_nodes * self.GNN_OUT

        # Flat state: GNN embedding + jobs_done + job_priority
        self.obs_dim = gnn_embed_size + num_jobs + num_jobs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_techs * num_jobs)

        # GNN encoder (shared, not trained via RL — used as feature extractor)
        self.gnn = GNNEncoder(self.NODE_FEAT, self.GNN_HIDDEN, self.GNN_OUT)
        self.gnn.eval()

        # Skill matrix: row = tech, col = job type (1 = capable)
        # job types: 0=Mechanical, 1=Electrical, 2=IT, 3=Civil
        self.skill_matrix = None   # set in reset()

    # ── Coordinate helpers ─────────────────────────────────────────────────
    def _rand_lat(self): return np.random.uniform(self.LAT_MIN, self.LAT_MAX)
    def _rand_lon(self): return np.random.uniform(self.LON_MIN, self.LON_MAX)

    def _norm_lat(self, lat): return (lat - self.LAT_MIN) / (self.LAT_MAX - self.LAT_MIN)
    def _norm_lon(self, lon): return (lon - self.LON_MIN) / (self.LON_MAX - self.LON_MIN)

    # ── Build adjacency matrix (fully connected, weighted by inverse dist) ─
    def _build_adjacency(self) -> torch.Tensor:
        all_lats = np.concatenate([self.tech_lat, self.job_lat])
        all_lons = np.concatenate([self.tech_lon, self.job_lon])
        N   = self.num_nodes
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = haversine(all_lats[i], all_lons[i],
                                  all_lats[j], all_lons[j])
                    adj[i, j] = 1.0 / (d + 1e-6)   # inverse distance weight
        return torch.tensor(adj, dtype=torch.float32)

    # ── Build node feature matrix ──────────────────────────────────────────
    def _build_node_features(self) -> torch.Tensor:
        feats = []

        # Technician nodes: [norm_lat, norm_lon, skill_norm, availability, type=0]
        for i in range(self.num_techs):
            feats.append([
                self._norm_lat(self.tech_lat[i]),
                self._norm_lon(self.tech_lon[i]),
                self.tech_skill[i] / 5.0,
                float(self.tech_available[i]),
                0.0   # node type: technician
            ])

        # Job nodes: [norm_lat, norm_lon, priority_norm, done_flag, type=1]
        for j in range(self.num_jobs):
            feats.append([
                self._norm_lat(self.job_lat[j]),
                self._norm_lon(self.job_lon[j]),
                self.job_priority[j] / 5.0,
                float(self.jobs_done[j]),
                1.0   # node type: job
            ])

        return torch.tensor(feats, dtype=torch.float32)

    # ── GNN embedding → observation ────────────────────────────────────────
    def _get_state(self) -> np.ndarray:
        with torch.no_grad():
            node_feat = self._build_node_features()
            adj       = self._build_adjacency()
            embedding = self.gnn(node_feat, adj).numpy()   # (num_nodes * GNN_OUT,)

        extra = np.concatenate([
            self.jobs_done,
            self.job_priority / 5.0
        ]).astype(np.float32)

        return np.concatenate([embedding, extra]).astype(np.float32)

    # ── Reset ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Technician attributes
        self.tech_lat       = np.array([self._rand_lat() for _ in range(self.num_techs)])
        self.tech_lon       = np.array([self._rand_lon() for _ in range(self.num_techs)])
        self.tech_skill     = np.random.uniform(1, 5, self.num_techs)        # 1–5
        self.tech_available = np.ones(self.num_techs, dtype=np.float32)
        self.tech_job_count = np.zeros(self.num_techs, dtype=np.float32)    # workload
        self.tech_job_type  = np.random.randint(0, 4, self.num_techs)       # 0–3

        # Job attributes
        self.job_lat       = np.array([self._rand_lat() for _ in range(self.num_jobs)])
        self.job_lon       = np.array([self._rand_lon() for _ in range(self.num_jobs)])
        self.job_priority  = np.random.randint(1, 6, self.num_jobs).astype(np.float32)
        self.jobs_done     = np.zeros(self.num_jobs, dtype=np.float32)
        self.job_type_req  = np.random.randint(0, 4, self.num_jobs)         # required skill type

        # Skill matrix: tech i can do job j if types match (or tech is generalist skill≥4)
        self.skill_matrix = np.array([
            [1 if (self.tech_job_type[i] == self.job_type_req[j]
                   or self.tech_skill[i] >= 4.0) else 0
             for j in range(self.num_jobs)]
            for i in range(self.num_techs)
        ])

        self.current_step = 0
        return self._get_state(), {}

    # ── Step ───────────────────────────────────────────────────────────────
    def step(self, action: int):
        self.current_step += 1

        tech_id = action // self.num_jobs
        job_id  = action % self.num_jobs

        # Clamp safety
        tech_id = min(tech_id, self.num_techs - 1)
        job_id  = min(job_id,  self.num_jobs  - 1)

        reward = -1.0   # step penalty

        if not self.tech_available[tech_id]:
            reward -= 15.0   # technician busy penalty

        elif self.skill_matrix[tech_id][job_id] == 0:
            reward -= 20.0   # skill mismatch

        elif self.jobs_done[job_id] == 1:
            reward -= 10.0   # job already done

        else:
            dist_km  = haversine(
                self.tech_lat[tech_id], self.tech_lon[tech_id],
                self.job_lat[job_id],   self.job_lon[job_id]
            )
            priority  = self.job_priority[job_id]

            # Workload fairness bonus: prefer less-loaded technicians
            avg_jobs  = self.tech_job_count.mean()
            fairness  = max(0, avg_jobs - self.tech_job_count[tech_id])

            # Reward: high priority + close distance + balanced workload
            reward += (-dist_km * 0.5) + (priority * 10.0) + (fairness * 5.0)

            # Update state
            self.tech_lat[tech_id]      = self.job_lat[job_id]
            self.tech_lon[tech_id]      = self.job_lon[job_id]
            self.tech_available[tech_id] = 0.0   # mark busy during job
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
            print(f"  Tech {i:02d} | ({self.tech_lat[i]:.4f}, {self.tech_lon[i]:.4f})"
                  f" | Skill {self.tech_skill[i]:.1f} | Jobs {int(self.tech_job_count[i])}"
                  f" | {status}")
        for j in range(self.num_jobs):
            done = "✅" if self.jobs_done[j] else "⏳"
            print(f"  Job  {j:02d} | ({self.job_lat[j]:.4f}, {self.job_lon[j]:.4f})"
                  f" | Priority {int(self.job_priority[j])} | {done}")

    def close(self):
        pass
