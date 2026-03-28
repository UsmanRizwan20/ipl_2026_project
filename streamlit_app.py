"""
IPL 2026 Interactive Dashboard (Streamlit Version)
================================
Run:  streamlit run streamlit_app.py

Requires predictions_2026.json (run ipl_2026_predictor.py first).
"""

import json
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(page_title="IPL 2026 Predictions", page_icon="🏆", layout="wide", initial_sidebar_state="collapsed")

# ─── Load predictions ───────────────────────────────────────────
JSON_FILE = "predictions_2026.json"
if not os.path.exists(JSON_FILE):
    st.error(
        "predictions_2026.json not found.\n"
        "Please run:  python ipl_2026_predictor.py  first."
    )
    st.stop()

with open(JSON_FILE) as f:
    data = json.load(f)

win_df    = pd.DataFrame(data["win_probabilities"])
bat_df    = pd.DataFrame(data["top_run_scorers"])
bowl_df   = pd.DataFrame(data["top_wicket_takers"])
sq_df     = pd.DataFrame(data["squad_strengths"])
bat25     = pd.DataFrame(data["player_batting"])
bowl25    = pd.DataFrame(data["player_bowling"])
cv_auc    = data["model_cv_auc"]

# ─── Team colours ───────────────────────────────────────────────
TEAM_COLORS = {
    "Chennai Super Kings":        "#F9CD1C",
    "Mumbai Indians":              "#004BA0",
    "Royal Challengers Bengaluru": "#D1001F",
    "Kolkata Knight Riders":       "#3A225D",
    "Delhi Capitals":              "#17449B",
    "Gujarat Titans":              "#1B2133",
    "Rajasthan Royals":            "#EA1A85",
    "Sunrisers Hyderabad":         "#F7A721",
    "Lucknow Super Giants":        "#A72056",
    "Punjab Kings":                "#AA2031",
}

SHORT = {
    "Chennai Super Kings":        "CSK",
    "Mumbai Indians":              "MI",
    "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders":       "KKR",
    "Delhi Capitals":              "DC",
    "Gujarat Titans":              "GT",
    "Rajasthan Royals":            "RR",
    "Sunrisers Hyderabad":         "SRH",
    "Lucknow Super Giants":        "LSG",
    "Punjab Kings":                "PBKS",
}

win_df["short"]  = win_df["team"].map(SHORT)
win_df["color"]  = win_df["team"].map(TEAM_COLORS)
sq_df["short"]   = sq_df["team"].map(SHORT)
sq_df["color"]   = sq_df["team"].map(TEAM_COLORS)

top_team   = win_df.iloc[0]
top_bat    = bat_df.iloc[0]
top_bowl   = bowl_df.iloc[0]

# ── Figures ──
def fig_win_prob():
    df = win_df.sort_values("win_probability", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["win_probability"],
        y=df["short"],
        orientation="h",
        marker_color=df["color"].tolist(),
        text=[f"{v:.1f}%" for v in df["win_probability"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Win Probability: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="🏆 Tournament Win Probability", font=dict(size=18)),
        xaxis=dict(title="Win Probability (%)", range=[0, df["win_probability"].max()*1.18]),
        yaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=60, t=60, b=40),
        height=400,
    )
    return fig

def fig_squad_strength():
    df = sq_df.sort_values("combined", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Batting", x=df["short"], y=df["bat"],
        marker_color=[TEAM_COLORS.get(t,"#888") for t in df["team"]],
        opacity=0.9,
        hovertemplate="<b>%{x}</b><br>Bat Strength: %{y:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Bowling", x=df["short"], y=df["bowl"],
        marker_color=[TEAM_COLORS.get(t,"#888") for t in df["team"]],
        opacity=0.55,
        hovertemplate="<b>%{x}</b><br>Bowl Strength: %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text="📊 2026 Squad Strength Breakdown", font=dict(size=18)),
        xaxis=dict(title=""),
        yaxis=dict(title="Strength Score"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
    )
    return fig

def fig_run_scorers(top_n=10):
    df = bat25.head(top_n).copy()
    df["short_team"] = df["team"].map(SHORT)
    fig = go.Figure(go.Bar(
        x=df["projected_runs_2026"],
        y=df["player"],
        orientation="h",
        marker_color=[TEAM_COLORS.get(t,"#888") for t in df["team"]],
        text=[f"~{v} runs" for v in df["projected_runs_2026"]],
        textposition="outside",
        customdata=df[["career_runs","short_team"]].values,
        hovertemplate="<b>%{y}</b> (%{customdata[1]})<br>"
                      "Projected 2026: %{x} runs<br>"
                      "Career IPL: %{customdata[0]:,.0f} runs<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="🏏 Predicted Top Run Scorers", font=dict(size=18)),
        xaxis=dict(title="Projected 2026 Runs", range=[0, df["projected_runs_2026"].max()*1.22]),
        yaxis=dict(title="", autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=140, r=80, t=60, b=40),
        height=420,
    )
    return fig

def fig_wicket_takers(top_n=10):
    df = bowl25.head(top_n).copy()
    df["short_team"] = df["team"].map(SHORT)
    fig = go.Figure(go.Bar(
        x=df["projected_wickets_2026"],
        y=df["player"],
        orientation="h",
        marker_color=[TEAM_COLORS.get(t,"#888") for t in df["team"]],
        text=[f"~{v} wkts" for v in df["projected_wickets_2026"]],
        textposition="outside",
        customdata=df[["career_wickets","short_team"]].values,
        hovertemplate="<b>%{y}</b> (%{customdata[1]})<br>"
                      "Projected 2026: %{x} wickets<br>"
                      "Career IPL: %{customdata[0]:.0f} wickets<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="🎳 Predicted Top Wicket Takers", font=dict(size=18)),
        xaxis=dict(title="Projected 2026 Wickets", range=[0, df["projected_wickets_2026"].max()*1.25]),
        yaxis=dict(title="", autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=140, r=80, t=60, b=40),
        height=420,
    )
    return fig

def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return "rgba(245,166,35,0.2)"

def fig_radar(team):
    """Radar chart comparing selected team vs all-team averages."""
    all_bat  = sq_df["bat"].mean()
    all_bowl = sq_df["bowl"].mean()
    all_comb = sq_df["combined"].mean()
    row      = sq_df[sq_df["team"]==team]
    if not len(row): return go.Figure()
    r = row.iloc[0]
    wp_row = win_df[win_df["team"]==team]
    wp     = wp_row.iloc[0]["win_probability"] if len(wp_row) else 10
    avg_wp = win_df["win_probability"].mean()

    cats   = ["Batting Strength","Bowling Strength","Combined Strength","Win Probability"]
    team_v = [r["bat"]/all_bat*10, r["bowl"]/all_bowl*10, r["combined"]/all_comb*10, wp/avg_wp*10]
    avg_v  = [10, 10, 10, 10]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_v + [avg_v[0]], theta=cats + [cats[0]],
        fill="toself", name="League Average",
        line_color="#555", fillcolor="rgba(85,85,85,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=team_v + [team_v[0]], theta=cats + [cats[0]],
        fill="toself", name=SHORT.get(team, team),
        line_color=TEAM_COLORS.get(team,"#f5a623"),
        fillcolor=hex_to_rgba(TEAM_COLORS.get(team,"#f5a623")),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,15], showticklabels=False, gridcolor="#333"),
            angularaxis=dict(gridcolor="#333"),
        ),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=f"{SHORT.get(team,team)} — Team Profile vs League Avg", font=dict(size=16)),
        margin=dict(l=50, r=50, t=70, b=50),
        height=360,
    )
    return fig

# ─── Streamlit UI Layout ────────────────────────────────────────

st.caption("◈ ML POWERED · 18 SEASONS · 278K+ DELIVERIES")
st.title("IPL 2026 PREDICTION ENGINE")
st.markdown("**Gradient Boosting model trained on 2008–2025 IPL data**")
st.divider()

# ── Summary cards ──
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🏆 Most Likely Champion",
        value=SHORT.get(top_team["team"], top_team["team"]),
        delta=f"{top_team['win_probability']:.1f}% prob",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="🏏 Top Run Scorer",
        value=top_bat["player"],
        delta=f"{int(top_bat['projected_runs_2026'])} proj runs",
        delta_color="off"
    )

with col3:
    st.metric(
        label="🎳 Top Wicket Taker",
        value=top_bowl["player"],
        delta=f"{int(top_bowl['projected_wickets_2026'])} proj wkts",
        delta_color="off"
    )

with col4:
    st.metric(
        label="🤖 Model Accuracy",
        value=f"{cv_auc * 100:.1f}%",
        delta="5-fold CV",
        delta_color="off"
    )

st.divider()

# ── Win probability + squad radar ──
row1_col1, row1_col2 = st.columns([1.5, 1])

with row1_col1:
    st.plotly_chart(fig_win_prob(), use_container_width=True)

with row1_col2:
    selected_team = st.selectbox(
        "Select Team Profile →",
        options=sorted(TEAM_COLORS.keys()),
        index=sorted(TEAM_COLORS.keys()).index("Mumbai Indians"),
        format_func=lambda x: f"{SHORT.get(x, x)} — {x}"
    )
    st.plotly_chart(fig_radar(selected_team), use_container_width=True)

st.divider()

# ── Squad strength ──
st.plotly_chart(fig_squad_strength(), use_container_width=True)

st.divider()

# ── Sliders + player charts ──
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    top_n_batters = st.slider("Show top N batters →", min_value=5, max_value=25, value=10, step=5)
    st.plotly_chart(fig_run_scorers(top_n_batters), use_container_width=True)

with row2_col2:
    top_n_bowlers = st.slider("Show top N bowlers →", min_value=5, max_value=25, value=10, step=5)
    st.plotly_chart(fig_wicket_takers(top_n_bowlers), use_container_width=True)

st.divider()

# ── Full data tables ──
st.subheader("📋 Full Predictions Data")

tab1, tab2, tab3, tab4 = st.tabs([
    "Win Probabilities",
    "Batting Leaderboard",
    "Bowling Leaderboard",
    "Squad Strengths"
])

with tab1:
    df_win = win_df[["team", "win_probability"]].copy()
    df_win["win_probability"] = df_win["win_probability"].round(2)
    df_win.columns = ["Team", "Win Probability (%)"]
    st.dataframe(df_win, use_container_width=True, hide_index=True)

with tab2:
    df_bat = bat25[["player", "team", "career_runs", "projected_runs_2026"]].copy()
    df_bat.columns = ["Player", "Team", "Career IPL Runs", "Projected 2026 Runs"]
    st.dataframe(df_bat, use_container_width=True, hide_index=True)

with tab3:
    df_bowl = bowl25[["player", "team", "career_wickets", "projected_wickets_2026"]].copy()
    df_bowl.columns = ["Player", "Team", "Career IPL Wickets", "Projected 2026 Wickets"]
    st.dataframe(df_bowl, use_container_width=True, hide_index=True)

with tab4:
    df_sq = sq_df[["team", "bat", "bowl", "combined"]].copy()
    df_sq[["bat", "bowl", "combined"]] = df_sq[["bat", "bowl", "combined"]].round(1)
    df_sq.columns = ["Team", "Batting Strength", "Bowling Strength", "Combined Strength"]
    df_sq = df_sq.sort_values("Combined Strength", ascending=False)
    st.dataframe(df_sq, use_container_width=True, hide_index=True)
