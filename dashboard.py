"""
IPL 2026 Interactive Dashboard
================================
Run:  python dashboard.py
Then open: http://127.0.0.1:8050

Requires predictions_2026.json (run ipl_2026_predictor.py first).
Extra dep: pip install dash plotly
"""

import json
import os
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─── Load predictions ───────────────────────────────────────────
JSON_FILE = "predictions_2026.json"
if not os.path.exists(JSON_FILE):
    raise FileNotFoundError(
        "predictions_2026.json not found.\n"
        "Please run:  python ipl_2026_predictor.py  first."
    )

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

# ─── App ────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="IPL 2026 Predictions")

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
        plot_bgcolor="#0f111a", paper_bgcolor="#0f111a",
        font=dict(color="#e8eaf0", family="DM Sans"),
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
        plot_bgcolor="#0f111a", paper_bgcolor="#0f111a",
        font=dict(color="#e8eaf0", family="DM Sans"),
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
        plot_bgcolor="#0f111a", paper_bgcolor="#0f111a",
        font=dict(color="#e8eaf0", family="DM Sans"),
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
        plot_bgcolor="#0f111a", paper_bgcolor="#0f111a",
        font=dict(color="#e8eaf0", family="DM Sans"),
        margin=dict(l=140, r=80, t=60, b=40),
        height=420,
    )
    return fig

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

    cats   = ["Batting<br>Strength","Bowling<br>Strength","Combined<br>Strength","Win<br>Probability"]
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
        fillcolor=TEAM_COLORS.get(team,"#f5a623").replace("#","rgba(") + ",0.2)" if "#" in TEAM_COLORS.get(team,"") else "rgba(245,166,35,0.2)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1a1d2e",
            radialaxis=dict(visible=True, range=[0,15], showticklabels=False, gridcolor="#333"),
            angularaxis=dict(gridcolor="#333"),
        ),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaf0")),
        paper_bgcolor="#0f111a",
        font=dict(color="#e8eaf0", family="DM Sans"),
        title=dict(text=f"{SHORT.get(team,team)} — Team Profile vs League Avg", font=dict(size=16)),
        margin=dict(l=50, r=50, t=70, b=50),
        height=360,
    )
    return fig

# ── Top summary cards ──
def summary_card(icon, label, value, sub, color):
    return html.Div([
        html.Div(icon, style={"fontSize":"28px","marginBottom":"6px"}),
        html.Div(label, style={"fontSize":"11px","letterSpacing":"2px","textTransform":"uppercase",
                                "color":"#7a7f94","marginBottom":"6px","fontFamily":"DM Mono, monospace"}),
        html.Div(value, style={"fontSize":"22px","fontWeight":"700","color":color,"marginBottom":"4px"}),
        html.Div(sub,   style={"fontSize":"13px","color":"#aab","lineHeight":"1.4"}),
    ], style={
        "background":"#141720","border":f"1px solid {color}33",
        "borderLeft":f"3px solid {color}",
        "borderRadius":"10px","padding":"20px","flex":"1","minWidth":"200px",
        "boxShadow":f"0 0 24px {color}15",
    })

top_team   = win_df.iloc[0]
top_bat    = bat_df.iloc[0]
top_bowl   = bowl_df.iloc[0]

# ── Layout ──
app.layout = html.Div([

    # ── Hero ──
    html.Div([
        html.Div([
            html.Span("◈ ML POWERED  ·  18 SEASONS  ·  278K+ DELIVERIES",
                      style={"fontFamily":"DM Mono, monospace","fontSize":"11px",
                             "letterSpacing":"3px","color":"#f5a623","display":"block","marginBottom":"14px"}),
            html.H1("IPL 2026", style={"fontFamily":"Bebas Neue, sans-serif","fontSize":"72px",
                                        "letterSpacing":"6px","margin":"0","lineHeight":"1",
                                        "background":"linear-gradient(135deg,#f5a623,#e8375a)",
                                        "-webkit-background-clip":"text","-webkit-text-fill-color":"transparent"}),
            html.H2("PREDICTION ENGINE", style={"fontFamily":"Bebas Neue, sans-serif","fontSize":"34px",
                                                  "letterSpacing":"10px","color":"#7a7f94",
                                                  "margin":"0","fontWeight":"400"}),
            html.P(f"Gradient Boosting model trained on 2008–2025 IPL data  ·  CV AUC {cv_auc:.3f}",
                   style={"color":"#7a7f94","fontSize":"13px","marginTop":"18px","fontFamily":"DM Mono, monospace"}),
        ], style={"maxWidth":"900px","margin":"0 auto"}),
    ], style={"padding":"52px 40px 40px","borderBottom":"1px solid rgba(255,255,255,0.07)",
              "background":"radial-gradient(ellipse 80% 60% at 70% -10%,rgba(245,166,35,0.10) 0%,transparent 70%)"}),

    # ── Summary cards ──
    html.Div([
        summary_card("🏆","Most Likely Champion",
                     SHORT.get(top_team["team"], top_team["team"]),
                     f"{top_team['win_probability']:.1f}% probability  ·  {top_team['team']}",
                     "#f5a623"),
        summary_card("🏏","Top Run Scorer",
                     top_bat["player"],
                     f"~{int(top_bat['projected_runs_2026'])} projected  ·  {SHORT.get(top_bat['team'],top_bat['team'])}",
                     "#4fc3f7"),
        summary_card("🎳","Top Wicket Taker",
                     top_bowl["player"],
                     f"~{int(top_bowl['projected_wickets_2026'])} projected  ·  {SHORT.get(top_bowl['team'],top_bowl['team'])}",
                     "#e8375a"),
        summary_card("🤖","Model Accuracy",
                     f"AUC {cv_auc:.3f}",
                     "5-fold cross-validation  ·  GradientBoosting",
                     "#a5d6a7"),
    ], style={"display":"flex","gap":"20px","padding":"30px 40px","flexWrap":"wrap"}),

    # ── Win probability + squad radar ──
    html.Div([
        html.Div([dcc.Graph(figure=fig_win_prob(), config={"displayModeBar":False})],
                 style={"flex":"1.4","minWidth":"340px"}),
        html.Div([
            html.Div([
                html.Label("Select Team Profile →",
                           style={"fontFamily":"DM Mono, monospace","fontSize":"11px",
                                  "letterSpacing":"2px","textTransform":"uppercase",
                                  "color":"#7a7f94","marginBottom":"10px","display":"block"}),
                dcc.Dropdown(
                    id="team-select",
                    options=[{"label":f"{SHORT.get(t,t)}  —  {t}","value":t} for t in sorted(TEAM_COLORS.keys())],
                    value="Mumbai Indians",
                    clearable=False,
                    style={"background":"#1e2130","border":"1px solid #333","color":"#000"},
                ),
            ], style={"padding":"20px 20px 0"}),
            dcc.Graph(id="radar-chart", config={"displayModeBar":False}),
        ], style={"flex":"1","minWidth":"320px","background":"#0f111a",
                  "border":"1px solid rgba(255,255,255,0.06)","borderRadius":"12px",
                  "margin":"0 0 0 0"}),
    ], style={"display":"flex","gap":"20px","padding":"0 40px 20px","flexWrap":"wrap"}),

    # ── Squad strength ──
    html.Div([
        dcc.Graph(figure=fig_squad_strength(), config={"displayModeBar":False}),
    ], style={"padding":"0 40px 20px"}),

    # ── Sliders + player charts ──
    html.Div([
        # Run scorers
        html.Div([
            html.Label("Show top N batters →",
                       style={"fontFamily":"DM Mono, monospace","fontSize":"11px",
                              "letterSpacing":"2px","textTransform":"uppercase",
                              "color":"#7a7f94","marginBottom":"10px","display":"block"}),
            dcc.Slider(id="bat-slider", min=5, max=25, step=5, value=10,
                       marks={i:str(i) for i in range(5,26,5)},
                       tooltip={"placement":"bottom"}),
            dcc.Graph(id="bat-chart", config={"displayModeBar":False}),
        ], style={"flex":"1","minWidth":"360px","background":"#0f111a",
                  "border":"1px solid rgba(255,255,255,0.06)","borderRadius":"12px","padding":"20px"}),

        # Wicket takers
        html.Div([
            html.Label("Show top N bowlers →",
                       style={"fontFamily":"DM Mono, monospace","fontSize":"11px",
                              "letterSpacing":"2px","textTransform":"uppercase",
                              "color":"#7a7f94","marginBottom":"10px","display":"block"}),
            dcc.Slider(id="bowl-slider", min=5, max=25, step=5, value=10,
                       marks={i:str(i) for i in range(5,26,5)},
                       tooltip={"placement":"bottom"}),
            dcc.Graph(id="bowl-chart", config={"displayModeBar":False}),
        ], style={"flex":"1","minWidth":"360px","background":"#0f111a",
                  "border":"1px solid rgba(255,255,255,0.06)","borderRadius":"12px","padding":"20px"}),
    ], style={"display":"flex","gap":"20px","padding":"0 40px 20px","flexWrap":"wrap"}),

    # ── Full data tables ──
    html.Div([
        html.H3("📋 Full Predictions Data",
                style={"fontFamily":"Bebas Neue, sans-serif","fontSize":"28px",
                       "letterSpacing":"4px","color":"#e8eaf0","marginBottom":"20px"}),
        dcc.Tabs(id="tabs", value="tab-win", children=[
            dcc.Tab(label="Win Probabilities", value="tab-win"),
            dcc.Tab(label="Batting Leaderboard", value="tab-bat"),
            dcc.Tab(label="Bowling Leaderboard", value="tab-bowl"),
            dcc.Tab(label="Squad Strengths", value="tab-sq"),
        ], colors={"border":"#333","primary":"#f5a623","background":"#141720"},
           style={"fontFamily":"DM Sans, sans-serif","fontSize":"14px"}),
        html.Div(id="tab-content", style={"marginTop":"16px"}),
    ], style={"padding":"10px 40px 60px"}),

], style={"background":"#07080d","minHeight":"100vh","color":"#e8eaf0",
          "fontFamily":"DM Sans, sans-serif"})

# ─── Callbacks ──────────────────────────────────────────────────
TABLE_STYLE = {
    "style_table": {"overflowX":"auto","borderRadius":"8px","overflow":"hidden"},
    "style_cell": {"backgroundColor":"#141720","color":"#e8eaf0","border":"1px solid #222",
                   "padding":"10px 14px","fontFamily":"DM Sans, sans-serif","fontSize":"14px"},
    "style_header": {"backgroundColor":"#1e2130","color":"#f5a623","fontWeight":"700",
                     "border":"1px solid #333","fontFamily":"DM Mono, monospace",
                     "fontSize":"12px","letterSpacing":"1px"},
    "style_data_conditional": [
        {"if":{"row_index":"odd"},"backgroundColor":"#0f111a"},
    ],
}

@app.callback(Output("radar-chart","figure"), Input("team-select","value"))
def update_radar(team): return fig_radar(team)

@app.callback(Output("bat-chart","figure"), Input("bat-slider","value"))
def update_bat(n): return fig_run_scorers(n)

@app.callback(Output("bowl-chart","figure"), Input("bowl-slider","value"))
def update_bowl(n): return fig_wicket_takers(n)

@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render_tab(tab):
    if tab == "tab-win":
        df = win_df[["team","win_probability"]].copy()
        df["win_probability"] = df["win_probability"].round(2)
        df.columns = ["Team","Win Probability (%)"]
        return dash_table.DataTable(data=df.to_dict("records"), columns=[{"name":c,"id":c} for c in df.columns], **TABLE_STYLE)
    elif tab == "tab-bat":
        df = bat25[["player","team","career_runs","projected_runs_2026"]].copy()
        df.columns = ["Player","Team","Career IPL Runs","Projected 2026 Runs"]
        return dash_table.DataTable(data=df.to_dict("records"), columns=[{"name":c,"id":c} for c in df.columns], **TABLE_STYLE)
    elif tab == "tab-bowl":
        df = bowl25[["player","team","career_wickets","projected_wickets_2026"]].copy()
        df.columns = ["Player","Team","Career IPL Wickets","Projected 2026 Wickets"]
        return dash_table.DataTable(data=df.to_dict("records"), columns=[{"name":c,"id":c} for c in df.columns], **TABLE_STYLE)
    elif tab == "tab-sq":
        df = sq_df[["team","bat","bowl","combined"]].copy()
        df[["bat","bowl","combined"]] = df[["bat","bowl","combined"]].round(1)
        df.columns = ["Team","Batting Strength","Bowling Strength","Combined Strength"]
        df = df.sort_values("Combined Strength", ascending=False)
        return dash_table.DataTable(data=df.to_dict("records"), columns=[{"name":c,"id":c} for c in df.columns], **TABLE_STYLE)


# ─── Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  IPL 2026 Dashboard → http://127.0.0.1:8050")
    print("=" * 50)
    app.run(debug=False, host="127.0.0.1", port=8050)
