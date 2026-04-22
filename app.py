"""
app.py  —  Streamlit Dashboard for Field Service Dispatch System
Reads assignments.csv written by FastAPI (called from n8n).
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
from streamlit_folium import st_folium
from datetime import datetime

st.set_page_config(
    page_title="Field Dispatch Dashboard — Kuantan",
    page_icon="🔧",
    layout="wide",
)

CENTER_LAT      = 3.8077
CENTER_LON      = 103.3260
RADIUS_KM       = 20
ASSIGNMENTS_CSV = "assignments.csv"
REFRESH_SEC     = 10

@st.cache_data(ttl=REFRESH_SEC)
def load_assignments():
    if os.path.exists(ASSIGNMENTS_CSV):
        df = pd.read_csv(ASSIGNMENTS_CSV)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp", ascending=False)
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_technicians():
    if os.path.exists("technician_dataset.csv"):
        return pd.read_csv("technician_dataset.csv")
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_workload():
    if os.path.exists("workload_dataset.csv"):
        return pd.read_csv("workload_dataset.csv")
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_supervision():
    if os.path.exists("supervision_dataset.csv"):
        return pd.read_csv("supervision_dataset.csv")
    return pd.DataFrame()

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🔧 Field Service Dispatch — Live Dashboard")
st.caption(
    f"Kuantan, Pahang · {RADIUS_KM} km radius from Padang MBK 1 · "
    f"Jobs submitted via n8n · Auto-refreshes every {REFRESH_SEC}s"
)

df_assign = load_assignments()
df_tech   = load_technicians()
df_wl     = load_workload()
df_sup    = load_supervision()

# ── KPI row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Jobs",        len(df_assign) if not df_assign.empty else 0)
c2.metric("Today",
          len(df_assign[df_assign["timestamp"].dt.date == datetime.today().date()])
          if not df_assign.empty else 0)
c3.metric("Avg ETA (min)",
          f"{df_assign['eta_min'].mean():.1f}" if not df_assign.empty else "—")
c4.metric("Avg Distance (km)",
          f"{df_assign['distance_km'].mean():.2f}" if not df_assign.empty else "—")
c5.metric("Techs Available",
          len(df_tech[df_tech["status"] == "Available"]) if not df_tech.empty else "—")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Live Map", "📋 Assignment Log", "📊 Workload Analysis", "📈 Model Evaluation"]
)

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Map
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    col_map, col_panel = st.columns([3, 1])

    with col_map:
        st.subheader("Live Technician & Job Locations")
        m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=12,
                       tiles="CartoDB positron")

        folium.Circle(
            [CENTER_LAT, CENTER_LON], radius=RADIUS_KM * 1000,
            color="#2563eb", fill=True, fill_opacity=0.04, weight=1.5,
            tooltip=f"{RADIUS_KM} km service radius"
        ).add_to(m)

        folium.Marker(
            [CENTER_LAT, CENTER_LON],
            popup="Padang MBK 1 (Reference)",
            icon=folium.Icon(color="black", icon="star", prefix="fa"),
        ).add_to(m)

        if not df_tech.empty:
            for _, t in df_tech.iterrows():
                colour = ("green"  if t["status"] == "Available" else
                          "orange" if t["status"] == "Working"   else "red")
                folium.Marker(
                    [t["lat"], t["lon"]],
                    popup=folium.Popup(
                        f"<b>{t['technician_id']}</b><br>"
                        f"{t['job_type']} · Skill {t['skill']}<br>{t['status']}",
                        max_width=180),
                    icon=folium.Icon(color=colour, icon="user", prefix="fa"),
                    tooltip=f"{t['technician_id']} — {t['status']}",
                ).add_to(m)

        if not df_assign.empty:
            for _, a in df_assign.head(20).iterrows():
                folium.Marker(
                    [a["job_lat"], a["job_lon"]],
                    popup=folium.Popup(
                        f"<b>{a['customer_name']}</b><br>"
                        f"P{a['job_priority']} · {a['required_skill']}<br>"
                        f"→ {a['assigned_to']} · ETA {a['eta_min']} min",
                        max_width=200),
                    icon=folium.Icon(color="red", icon="wrench", prefix="fa"),
                    tooltip=f"Job: {a['customer_name']}",
                ).add_to(m)

                if pd.notna(a.get("tech_lat")) and pd.notna(a.get("tech_lon")):
                    folium.PolyLine(
                        [[a["tech_lat"], a["tech_lon"]], [a["job_lat"], a["job_lon"]]],
                        color="#16a34a", weight=2, dash_array="6",
                        tooltip=f"{a['assigned_to']} → {a['customer_name']}"
                    ).add_to(m)

        st_folium(m, height=520, use_container_width=True)

    with col_panel:
        st.subheader("Technician Status")
        if not df_tech.empty:
            for _, t in df_tech.iterrows():
                dot = ("🟢" if t["status"] == "Available" else
                       "🟡" if t["status"] == "Working"   else "🔴")
                st.markdown(f"{dot} **{t['technician_id']}**  \n"
                            f"&nbsp;&nbsp;{t['job_type']} · Skill {t['skill']}")
        else:
            st.info("Run generate_data.py")

        st.divider()
        st.subheader("Recent Dispatches")
        if not df_assign.empty:
            for _, a in df_assign.head(5).iterrows():
                st.markdown(
                    f"🔧 **{a['assigned_to']}** → {a['customer_name']}  \n"
                    f"&nbsp;&nbsp;P{a['job_priority']} · {a['eta_min']} min  \n"
                    f"&nbsp;&nbsp;`{str(a['timestamp'])[:16]}`"
                )
        else:
            st.info("No assignments yet.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Assignment Log
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📋 All Assignments (submitted via n8n)")

    if df_assign.empty:
        st.info("No assignments yet. Submit a job through the n8n form.")
    else:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            priority_filter = st.multiselect("Priority", [1,2,3,4,5], default=[1,2,3,4,5])
        with fc2:
            skills = df_assign["required_skill"].unique().tolist() if "required_skill" in df_assign else []
            skill_filter = st.multiselect("Skill Type", skills, default=skills)
        with fc3:
            search = st.text_input("Search customer / technician")

        filtered = df_assign[df_assign["job_priority"].isin(priority_filter)]
        if skill_filter:
            filtered = filtered[filtered["required_skill"].isin(skill_filter)]
        if search:
            mask = (
                filtered["customer_name"].str.contains(search, case=False, na=False) |
                filtered["assigned_to"].str.contains(search,  case=False, na=False)
            )
            filtered = filtered[mask]

        st.dataframe(
            filtered[["timestamp","customer_name","address","problem",
                       "job_priority","required_skill","assigned_to",
                       "distance_km","eta_min","status"]].rename(columns={
                "timestamp":"Time","customer_name":"Customer","address":"Address",
                "problem":"Problem","job_priority":"Priority",
                "required_skill":"Skill","assigned_to":"Technician",
                "distance_km":"Dist (km)","eta_min":"ETA (min)","status":"Status",
            }),
            use_container_width=True, height=400,
        )

        st.download_button("⬇️ Download CSV",
                           data=filtered.to_csv(index=False),
                           file_name="assignments_export.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Workload Analysis
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Workload Distribution")

    if not df_assign.empty:
        jobs_per_tech = (df_assign.groupby("assigned_to").size()
                         .reset_index(name="jobs_assigned")
                         .sort_values("jobs_assigned", ascending=False))
        avg_dist = (df_assign.groupby("assigned_to")["distance_km"]
                    .mean().reset_index().rename(columns={"distance_km":"avg_dist_km"}))

        wc1, wc2 = st.columns(2)
        with wc1:
            st.markdown("#### Jobs Assigned per Technician")
            st.bar_chart(jobs_per_tech.set_index("assigned_to")["jobs_assigned"])
        with wc2:
            st.markdown("#### Average Travel Distance per Technician (km)")
            st.bar_chart(avg_dist.set_index("assigned_to")["avg_dist_km"])

        st.markdown("#### Job Priority Distribution")
        st.bar_chart(df_assign["job_priority"].value_counts().sort_index())

        std_jobs = jobs_per_tech["jobs_assigned"].std()
        st.metric("Workload Fairness (Std Dev)", f"{std_jobs:.2f}",
                  delta="Lower = fairer", delta_color="inverse")

    elif not df_wl.empty:
        st.markdown("#### Jobs per Technician (Simulated Dataset)")
        st.bar_chart(df_wl.set_index("tech_id")["no_of_jobs"])
        if not df_sup.empty:
            st.markdown("#### Customer Wait Time Distribution (min)")
            st.bar_chart(df_sup["cust_wait_time"].round(0).value_counts().sort_index())
    else:
        st.info("Submit jobs via n8n or run generate_data.py to see charts.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Evaluation
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 Model Evaluation")
    st.markdown("""
    **Regret Analysis** *(Eq 3.7)*: `R(π) = J(π_ref) − J(π)` — Negative = outperforms baseline.

    **Bellman Consistency** *(Eq 3.8)*: `δt = rt + γ·V(st+1) − V(st)` — Lower |δt| = more stable.
    """)

    if os.path.exists("eval_log.csv"):
        eval_df = pd.read_csv("eval_log.csv")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("#### Regret — Baseline vs Learned Policy")
            st.line_chart(eval_df.set_index("episode")[["J_ref","J_learned"]])
        with ec2:
            st.markdown("#### Bellman Residual Convergence")
            st.line_chart(eval_df.set_index("episode")[["bellman_residual"]])

        final = eval_df.iloc[-1]
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Final Regret",           f"{final['regret']:.2f}")
        mc2.metric("Final Bellman Residual",  f"{final['bellman_residual']:.4f}")
        mc3.metric("Learned Policy Reward",   f"{final['J_learned']:.2f}")
    else:
        st.info("Run `python train.py` first to generate eval_log.csv.")

    if not df_assign.empty:
        st.divider()
        st.markdown("#### Live System Performance (n8n submissions)")
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Avg ETA",          f"{df_assign['eta_min'].mean():.1f} min")
        lc2.metric("Avg Distance",     f"{df_assign['distance_km'].mean():.2f} km")
        lc3.metric("Total Dispatches", len(df_assign))
        st.markdown("#### ETA Trend")
        st.line_chart(df_assign.sort_values("timestamp").set_index("timestamp")["eta_min"])

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
col_f1, col_f2 = st.columns([4, 1])
with col_f2:
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')} · Cache TTL {REFRESH_SEC}s")
