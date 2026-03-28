"""
IPL 2026 Win Probability & Player Performance Predictor
========================================================
Run:  python ipl_2026_predictor.py
Output: console results + predictions_2026.json

Data required in data/ folder:
  - ball_by_ball_data.csv
  - ipl_matches_data.csv
  - players-data-updated.csv
  - teams_data.csv
"""

import pandas as pd
import numpy as np
import warnings
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# DATA PATHS
# ──────────────────────────────────────────────
DATA_DIR     = "data"
MATCHES_PATH = f"{DATA_DIR}/ipl_matches_data.csv"
BBB_PATH     = f"{DATA_DIR}/ball_by_ball_data.csv"
PLAYERS_PATH = f"{DATA_DIR}/players-data-updated.csv"
TEAMS_PATH   = f"{DATA_DIR}/teams_data.csv"

# ──────────────────────────────────────────────
# IPL 2026 SQUAD DEFINITIONS
# ──────────────────────────────────────────────
SQUADS_2026 = {
    "Chennai Super Kings": [
        "Ruturaj Gaikwad","MS Dhoni","Sanju Samson","Shivam Dube",
        "Shreyas Gopal","Khaleel Ahmed","Ayush Mhatre","Urvil Patel",
        "Kartik Sharma","Prashant Veer","Rahul Chahar","Sarfaraz Khan",
        "Aman Khan","Anshul Kamboj","Gurjapneet Singh","Mukesh Choudhary",
        "Ramakrishna Ghosh","Jamie Overton","Spencer Johnson","Noor Ahmad",
        "Dewald Brevis","Matt Henry","Akeal Hosein","Matthew Short","Zakary Foulkes"
    ],
    "Mumbai Indians": [
        "Hardik Pandya","Rohit Sharma","Suryakumar Yadav","Jasprit Bumrah",
        "Tilak Varma","Robin Minz","Naman Dhir","Ashwani Kumar",
        "Deepak Chahar","Raghu Sharma","Raj Angad Bawa","Mayank Markande",
        "Shardul Thakur","Mohammad Izhar","Danish Malewar","Atharva Ankolekar",
        "Mayank Rawat","Quinton de Kock","Trent Boult","Mitchell Santner",
        "Corbin Bosch","Ryan Rickelton","Allah Ghazanfar","Will Jacks","Sherfane Rutherford"
    ],
    "Kolkata Knight Riders": [
        "Ajinkya Rahane","Rinku Singh","Angkrish Raghuvanshi","Manish Pandey",
        "Ramandeep Singh","Anukul Roy","Varun Chakravarthy","Harshit Rana",
        "Vaibhav Arora","Umran Malik","Tejasvi Singh","Saurabh Dubey",
        "Rahul Tripathi","Daksh Kamra","Sarthak Ranjan","Prashant Solanki",
        "Kartik Tyagi","Akash Deep","Sunil Narine","Cameron Green",
        "Matheesha Pathirana","Rovman Powell","Finn Allen","Tim Seifert",
        "Rachin Ravindra","Mustafizur Rahman","Blessing Muzarabani"
    ],
    "Royal Challengers Bengaluru": [
        "Rajat Patidar","Virat Kohli","Devdutt Padikkal","Jitesh Sharma",
        "Krunal Pandya","Yash Dayal","Bhuvneshwar Kumar","Rasikh Dar",
        "Abhinandan Singh","Suyash Sharma","Swapnil Singh","Mangesh Yadav",
        "Satvik Deswal","Vicky Ostwal","Vihaan Malhotra","Kanishk Chouhan",
        "Phil Salt","Josh Hazlewood","Tim David","Romario Shepherd",
        "Nuwan Thushara","Jacob Bethell","Venkatesh Iyer","Jacob Duffy","Jordan Cox"
    ],
    "Delhi Capitals": [
        "Axar Patel","KL Rahul","Karun Nair","Abhishek Porel",
        "Sameer Rizvi","Ashutosh Sharma","T Natarajan","Mukesh Kumar",
        "Kuldeep Yadav","Nitish Rana","Madhav Tiwari","Tripurana Vijay",
        "Vipraj Nigam","Auqib Dar","Prithvi Shaw","Sahil Parakh",
        "Ajay Mandal","Mitchell Starc","David Miller","Tristan Stubbs",
        "Pathum Nissanka","Dushmantha Chameera","Lungi Ngidi","Kyle Jamieson","Ben Duckett"
    ],
    "Gujarat Titans": [
        "Shubman Gill","Sai Sudharsan","Kumar Kushagra","Anuj Rawat",
        "Nishant Sindhu","Washington Sundar","Arshad Khan","Shahrukh Khan",
        "Rahul Tewatia","Mohammed Siraj","Prasidh Krishna","Ishant Sharma",
        "Gurnoor Singh Brar","Manav Suthar","Sai Kishore","Jayant Yadav",
        "Ashok Sharma","Prithviraj Yarra","Jos Buttler","Kagiso Rabada",
        "Rashid Khan","Jason Holder","Glenn Phillips","Tom Banton","Luke Wood"
    ],
    "Rajasthan Royals": [
        "Riyan Parag","Yashasvi Jaiswal","Shubham Dubey","Vaibhav Suryavanshi",
        "Ravindra Jadeja","Dhruv Jurel","Ravi Bishnoi","Tushar Deshpande",
        "Yudhvir Singh Charak","Sushant Mishra","Yash Raj Punia",
        "Vignesh Puthur","Brijesh Sharma","Kuldeep Sen","Sandeep Sharma",
        "Aman Rao","Jofra Archer","Sam Curran","Shimron Hetmyer",
        "Kwena Maphaka","Nandre Burger","Donovan Ferreira","Lhuan-dre Pretorius","Adam Milne"
    ],
    "Sunrisers Hyderabad": [
        "Pat Cummins","Abhishek Sharma","Nitish Kumar Reddy","Ishan Kishan",
        "Harshal Patel","Jaydev Unadkat","Aniket Verma","Harsh Dubey",
        "R Smaran","Zeeshan Ansari","Salil Arora","Shivam Mavi",
        "Shivang Kumar","Praful Hinge","Amit Kumar","Onkar Tarmale",
        "Heinrich Klaasen","Travis Head","Liam Livingstone","Brydon Carse",
        "Kamindu Mendis","Jack Edwards","Sakib Hussain"
    ],
    "Lucknow Super Giants": [
        "Rishabh Pant","Mayank Yadav","Abdul Samad","Shahbaz Ahmed",
        "Ayush Badoni","Arshin Kulkarni","Avesh Khan","M Siddharth",
        "Digvesh Singh","Akash Singh","Prince Yadav","Arjun Tendulkar",
        "Naman Tiwari","Mohsin Khan","Mukul Choudhary","Akshat Raghuwanshi",
        "Himmat Singh","Nicholas Pooran","Aiden Markram","Mitchell Marsh",
        "Wanindu Hasaranga","Anrich Nortje","Matthew Breetzke","Josh Inglis"
    ],
    "Punjab Kings": [
        "Shreyas Iyer","Prabhsimran Singh","Shashank Singh","Arshdeep Singh",
        "Yuzvendra Chahal","Harpreet Brar","Nehal Wadhera","Vishnu Vinod",
        "Harnoor Pannu","Pyla Avinash","Priyansh Arya","Musheer Khan",
        "Suryansh Shedge","Vyshak Vijaykumar","Yash Thakur","Pravin Dubey",
        "Vishal Nishad","Marcus Stoinis","Marco Jansen","Azmatullah Omarzai",
        "Lockie Ferguson","Mitch Owen","Xavier Bartlett","Ben Dwarshuis","Cooper Connolly"
    ]
}

PLAYER_ALIASES = {
    "V Kohli":"Virat Kohli","RG Sharma":"Rohit Sharma",
    "MS Dhoni†":"MS Dhoni","MSD":"MS Dhoni",
    "JJ Bumrah":"Jasprit Bumrah","HH Pandya":"Hardik Pandya",
    "KL Rahul†":"KL Rahul","RA Jadeja":"Ravindra Jadeja",
    "YK Patidar":"Rajat Patidar","SV Samson":"Sanju Samson","SV Samson†":"Sanju Samson",
    "YBK Jaiswal":"Yashasvi Jaiswal","SH Gill":"Shubman Gill",
    "SP Narine":"Sunil Narine","Q de Kock†":"Quinton de Kock",
    "RK Singh":"Rinku Singh","RD Gaikwad":"Ruturaj Gaikwad",
    "TA Boult":"Trent Boult","TR Head":"Travis Head",
    "HE Klaasen†":"Heinrich Klaasen","NM Rana":"Nitish Rana",
    "NTK Reddy":"Nitish Kumar Reddy","SA Yadav":"Suryakumar Yadav",
    "DA Miller":"David Miller","RP Singh":"Riyan Parag",
    "SN Thakur":"Shardul Thakur","YS Chahal":"Yuzvendra Chahal",
    "K Rabada":"Kagiso Rabada","W Sundar":"Washington Sundar",
    "B Kumar":"Bhuvneshwar Kumar","KC Yadav":"Kuldeep Yadav",
    "AR Patel":"Axar Patel","KH Pandya":"Krunal Pandya",
    "DP Padikkal":"Devdutt Padikkal","HV Patel":"Harshal Patel",
    "N Varma":"Tilak Varma","PJ Cummins":"Pat Cummins",
    "JC Buttler†":"Jos Buttler","RR Pant†":"Rishabh Pant",
    "Ishan Kishan†":"Ishan Kishan","MJ Santner":"Mitchell Santner",
    "B Sai Sudharsan":"Sai Sudharsan","LS Livingstone":"Liam Livingstone",
    "SM Curran":"Sam Curran","JC Archer":"Jofra Archer",
    "SH Dube":"Shivam Dube","DL Chahar":"Deepak Chahar",
    "C Green":"Cameron Green","K Chakravarthy":"Varun Chakravarthy",
    "M Pathirana":"Matheesha Pathirana","MP Stoinis":"Marcus Stoinis",
    "MR Marsh":"Mitchell Marsh","AK Markram":"Aiden Markram",
    "A Nortje":"Anrich Nortje","N Pooran†":"Nicholas Pooran",
    "JR Hazlewood":"Josh Hazlewood","P Krishna":"Prasidh Krishna",
    "RA Tewatia":"Rahul Tewatia","GJ Maxwell":"Glenn Maxwell",
    "T Natarajan":"T Natarajan",
}

TEAM_NAME_MAP = {
    "Royal Challengers Bangalore":"Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru":"Royal Challengers Bengaluru",
    "Mumbai Indians":"Mumbai Indians",
    "Chennai Super Kings":"Chennai Super Kings",
    "Kolkata Knight Riders":"Kolkata Knight Riders",
    "Sunrisers Hyderabad":"Sunrisers Hyderabad",
    "Delhi Capitals":"Delhi Capitals","Delhi Daredevils":"Delhi Capitals",
    "Punjab Kings":"Punjab Kings","Kings XI Punjab":"Punjab Kings",
    "Rajasthan Royals":"Rajasthan Royals",
    "Gujarat Titans":"Gujarat Titans",
    "Lucknow Super Giants":"Lucknow Super Giants",
    "Deccan Chargers":"Sunrisers Hyderabad",
    "Rising Pune Supergiant":None,"Rising Pune Supergiants":None,
    "Gujarat Lions":None,"Pune Warriors":None,"Kochi Tuskers Kerala":None,
}

CURRENT_TEAMS = list(SQUADS_2026.keys())
SEASON_ORDER  = {
    "2008":1,"2009":2,"2010":3,"2011":4,"2012":5,"2013":6,
    "2014":7,"2015":8,"2016":9,"2017":10,"2018":11,"2019":12,
    "2020/21":13,"2021":14,"2022":15,"2023":16,"2024":17,"2025":18
}

# ──────────────────────────────────────────────
def normalize_name(name):
    if pd.isna(name): return name
    n = str(name).strip()
    return PLAYER_ALIASES.get(n, n)

def recency_weighted_agg(df, group_col, value_cols, season_col="season_num", max_s=18, half_life=3):
    df = df.copy()
    df["weight"] = 0.5 ** ((max_s - df[season_col]) / half_life)
    rows = []
    for player, grp in df.groupby(group_col):
        row = {"player": player}
        tw = grp["weight"].sum()
        for col in value_cols:
            row[f"w_{col}"] = (grp[col] * grp["weight"]).sum() / tw if tw > 0 else 0
        row["seasons_played"]    = len(grp)
        row[f"career_{value_cols[0]}"] = grp[value_cols[0]].sum()
        rows.append(row)
    return pd.DataFrame(rows)

# ──────────────────────────────────────────────
def run():
    print("=" * 62)
    print("  IPL 2026 PREDICTION ENGINE  |  18 Seasons of Data")
    print("=" * 62)

    # 1. Load
    print("\n[1/6] Loading data...")
    matches = pd.read_csv(MATCHES_PATH)
    bbb     = pd.read_csv(BBB_PATH)
    print(f"  Matches    : {len(matches):,}  across {matches['season'].nunique()} seasons")
    print(f"  Deliveries : {len(bbb):,}")

    matches["match_date"]        = pd.to_datetime(matches["match_date"], format="%d-%m-%Y", dayfirst=True)
    matches["team1_norm"]        = matches["team1"].map(lambda x: TEAM_NAME_MAP.get(x, x))
    matches["team2_norm"]        = matches["team2"].map(lambda x: TEAM_NAME_MAP.get(x, x))
    matches["winner_norm"]       = matches["match_winner"].map(lambda x: TEAM_NAME_MAP.get(x, x) if pd.notna(x) else x)
    matches["toss_winner_norm"]  = matches["toss_winner"].map(lambda x: TEAM_NAME_MAP.get(x, x))

    matches_cur = matches[
        matches["team1_norm"].isin(CURRENT_TEAMS) &
        matches["team2_norm"].isin(CURRENT_TEAMS)
    ].copy()

    # 2. Champions
    print("\n[2/6] Identifying season champions...")
    tournament_wins = {t: 0 for t in CURRENT_TEAMS}
    champions = {}
    for season, sg in matches.groupby("season"):
        last   = sg.sort_values("match_date").iloc[-1]
        winner = TEAM_NAME_MAP.get(last["match_winner"], last["match_winner"])
        if winner in CURRENT_TEAMS:
            champions[season] = winner
            tournament_wins[winner] += 1
    print(f"  Seasons with champion identified: {len(champions)}")
    for s, w in sorted(champions.items()):
        print(f"    {s}: {w}")
    print(f"  Title tally: { {k:v for k,v in tournament_wins.items() if v>0} }")

    # 3. Team season stats
    print("\n[3/6] Computing team season features...")
    records = []
    for season, sg in matches_cur.groupby("season"):
        for team in CURRENT_TEAMS:
            played = sg[(sg["team1_norm"]==team)|(sg["team2_norm"]==team)]
            if not len(played): continue
            wins   = played[played["winner_norm"]==team]
            tw_    = played[played["toss_winner_norm"]==team]
            tw_win = tw_[tw_["winner_norm"]==team]
            records.append({
                "season": season, "team": team,
                "played": len(played), "wins": len(wins),
                "win_rate": len(wins)/len(played),
                "toss_win_rate": len(tw_win)/len(tw_) if len(tw_)>0 else 0,
                "season_num": SEASON_ORDER.get(season, 0),
            })
    team_stats = pd.DataFrame(records)

    playoff_apps = {t: 0 for t in CURRENT_TEAMS}
    for _, row in matches_cur.iterrows():
        stage = str(row.get("stage",""))
        if any(s.lower() in stage.lower() for s in ["final","qualifier","eliminator","semi"]):
            for t in [row.get("team1_norm"), row.get("team2_norm")]:
                if t in playoff_apps: playoff_apps[t] += 1

    # 4. Player features
    print("\n[4/6] Engineering player features (recency-weighted)...")
    bbb["batter_norm"] = bbb["batter"].apply(normalize_name)
    bbb["bowler_norm"] = bbb["bowler"].apply(normalize_name)
    bbb_m    = bbb.merge(matches[["match_id","season"]].drop_duplicates(), on="match_id", how="left")
    bbb_m["season_num"] = bbb_m["season"].map(SEASON_ORDER)
    bbb_main = bbb_m[bbb_m["is_super_over"] == False].copy()

    # Batting aggregates
    bat_s = bbb_main[bbb_main["is_wide_ball"]==False].groupby(
        ["batter_norm","season","season_num"]
    ).agg(runs=("batter_runs","sum"), balls=("batter_runs","count")).reset_index()
    bat_s["strike_rate"] = bat_s["runs"] / bat_s["balls"].clip(1) * 100
    inn = bbb_main[bbb_main["is_wide_ball"]==False].groupby(
        ["batter_norm","season"])["match_id"].nunique().reset_index(name="innings")
    bat_s = bat_s.merge(inn, on=["batter_norm","season"], how="left")
    bat_s["avg"] = bat_s["runs"] / bat_s["innings"].clip(1)
    bat_agg = recency_weighted_agg(bat_s, "batter_norm", ["runs","strike_rate","avg"])

    # Bowling aggregates
    VALID_WK = {"bowled","caught","lbw","stumped","caught and bowled","hit wicket"}
    bowl_s = bbb_main[bbb_main["is_wide_ball"]==False].groupby(
        ["bowler_norm","season","season_num"]
    ).agg(balls=("total_runs","count"), runs_given=("total_runs","sum")).reset_index()
    wkts = bbb_main[
        (bbb_main["is_wicket"]==True) &
        (bbb_main["wicket_kind"].astype(str).str.lower().isin(VALID_WK))
    ].groupby(["bowler_norm","season"]).size().reset_index(name="wickets")
    bowl_s = bowl_s.merge(wkts, on=["bowler_norm","season"], how="left")
    bowl_s["wickets"] = bowl_s["wickets"].fillna(0).astype(int)
    bowl_s["overs"]   = bowl_s["balls"] / 6
    bowl_s["economy"] = bowl_s["runs_given"] / bowl_s["overs"].clip(0.1)
    bowl_agg = recency_weighted_agg(bowl_s, "bowler_norm", ["wickets","economy"])

    # 5. Squad scoring
    print("\n[5/6] Scoring 2026 squads...")

    def bat_score(p):
        cands = [p] + [k for k,v in PLAYER_ALIASES.items() if v.lower()==p.lower()]
        for c in cands:
            row = bat_agg[bat_agg["player"].str.lower()==c.lower()]
            if len(row):
                r  = row.iloc[0]
                sc = r["w_runs"]*1.0 + r["w_strike_rate"]*0.3 + r["w_avg"]*0.5 + np.sqrt(r.get("career_runs",0))*0.2
                return float(sc), float(r.get("career_runs",0)), float(r.get("w_runs",0))
        return 0.0, 0.0, 0.0

    def bowl_score(p):
        cands = [p] + [k for k,v in PLAYER_ALIASES.items() if v.lower()==p.lower()]
        for c in cands:
            row = bowl_agg[bowl_agg["player"].str.lower()==c.lower()]
            if len(row):
                r  = row.iloc[0]
                sc = r["w_wickets"]*5.0 + max(0,12-r["w_economy"])*2 + np.sqrt(r.get("career_wickets",0))*1.0
                return float(sc), float(r.get("career_wickets",0)), float(r.get("w_wickets",0))
        return 0.0, 0.0, 0.0

    squad_strength = {}
    pb_all, bw_all = {}, {}
    for team, squad in SQUADS_2026.items():
        pb = {p: bat_score(p)  for p in squad}
        bw = {p: bowl_score(p) for p in squad}
        pb_all[team], bw_all[team] = pb, bw
        top_bat  = sorted([v[0] for v in pb.values()], reverse=True)[:11]
        top_bowl = sorted([v[0] for v in bw.values()], reverse=True)[:11]
        squad_strength[team] = {
            "bat":  sum(top_bat), "bowl": sum(top_bowl),
            "combined": sum(top_bat)*0.5 + sum(top_bowl)*0.5
        }

    # 6. Model
    print("\n[6/6] Training win probability model...")
    champ_df = pd.DataFrame([{"season":s,"champion":c} for s,c in champions.items()])
    feat     = team_stats.merge(champ_df, on="season", how="left")
    feat["is_champion"]   = (feat["team"]==feat["champion"]).astype(int)
    feat["hist_titles"]   = feat["team"].map(tournament_wins).fillna(0)
    feat["hist_playoffs"] = feat["team"].map(playoff_apps).fillna(0)
    feat["bat_strength"]  = feat["team"].map(lambda t: squad_strength.get(t,{}).get("bat",0))
    feat["bowl_strength"] = feat["team"].map(lambda t: squad_strength.get(t,{}).get("bowl",0))
    feat = feat[feat["played"]>=5]

    FC = ["win_rate","toss_win_rate","played","hist_titles","hist_playoffs",
          "bat_strength","bowl_strength","season_num"]
    X = feat[FC].fillna(0).values
    y = feat["is_champion"].values

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    clf    = GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,
                                        max_depth=4,subsample=0.8,random_state=42)
    clf.fit(Xs, y)
    cv_auc = cross_val_score(clf, Xs, y, cv=5, scoring="roc_auc").mean()
    print(f"  Model CV AUC: {cv_auc:.3f}")

    # Predict 2026
    pred_rows = []
    for team in CURRENT_TEAMS:
        last3 = team_stats[team_stats["team"]==team].nlargest(3,"season_num")
        wr  = last3["win_rate"].mean()      if len(last3) else 0.5
        twr = last3["toss_win_rate"].mean() if len(last3) else 0.5
        pl  = last3["played"].mean()        if len(last3) else 10
        pred_rows.append({
            "team":team,"win_rate":wr,"toss_win_rate":twr,"played":pl,
            "hist_titles":tournament_wins.get(team,0),
            "hist_playoffs":playoff_apps.get(team,0),
            "bat_strength":squad_strength[team]["bat"],
            "bowl_strength":squad_strength[team]["bowl"],
            "season_num":19,
        })
    pred_df   = pd.DataFrame(pred_rows)
    raw_probs = clf.predict_proba(scaler.transform(pred_df[FC].fillna(0).values))[:,1]

    # Calibrated blend
    str_vals  = np.array([squad_strength[t]["combined"] for t in pred_df["team"]])
    str_p     = str_vals / str_vals.sum()
    hist_vals = np.array([tournament_wins.get(t,0) for t in pred_df["team"]], dtype=float)
    hist_p    = (hist_vals+0.5) / (hist_vals+0.5).sum()
    rec_wr    = np.array([
        team_stats[team_stats["team"]==t].nlargest(2,"season_num")["win_rate"].mean()
        if len(team_stats[team_stats["team"]==t]) else 0.4
        for t in pred_df["team"]
    ])
    rec_p     = rec_wr / rec_wr.sum()
    model_p   = raw_probs / raw_probs.sum() if raw_probs.sum()>0 else np.ones(10)/10

    final     = 0.40*str_p + 0.25*model_p + 0.20*hist_p + 0.15*rec_p
    pred_df["win_probability"] = final/final.sum()*100
    pred_df   = pred_df.sort_values("win_probability",ascending=False).reset_index(drop=True)

    # Player leaderboards
    bat_rows, bowl_rows = [], []
    for team, squad in SQUADS_2026.items():
        for p in squad:
            bs,cr,wr_ = pb_all[team][p]
            ws,cw,ww  = bw_all[team][p]
            bat_rows.append({"player":p,"team":team,"score":bs,"career_runs":cr,"w_runs":wr_})
            bowl_rows.append({"player":p,"team":team,"score":ws,"career_wickets":cw,"w_wickets":ww})
    bat_lb  = pd.DataFrame(bat_rows).sort_values("score",ascending=False)
    bat_lb  = bat_lb[bat_lb["score"]>0].reset_index(drop=True)
    bowl_lb = pd.DataFrame(bowl_rows).sort_values("score",ascending=False)
    bowl_lb = bowl_lb[bowl_lb["score"]>0].reset_index(drop=True)
    bat_lb["projected_runs_2026"]     = bat_lb["w_runs"].round(0).astype(int)
    bowl_lb["projected_wickets_2026"] = bowl_lb["w_wickets"].round(0).astype(int)

    # ── Print Results ──
    print("\n" + "="*62)
    print("  RESULTS: IPL 2026 PREDICTIONS")
    print("="*62)

    print("\n🏆  TOURNAMENT WIN PROBABILITY")
    print("─"*62)
    for _, row in pred_df.iterrows():
        bar = "█"*int(row["win_probability"]/100*35) + "░"*(35-int(row["win_probability"]/100*35))
        print(f"  {_+1:2}. {row['team']:<35}  {row['win_probability']:5.1f}%  {bar}")

    print("\n🏏  TOP 10 RUN SCORERS  (2026 squads only)")
    print("─"*62)
    for i, row in bat_lb.head(10).iterrows():
        print(f"  {i+1:2}. {row['player']:<25} ({row['team']:<35}) "
              f" Career: {int(row['career_runs']):,}  Proj: ~{row['projected_runs_2026']}")

    print("\n🎳  TOP 10 WICKET TAKERS  (2026 squads only)")
    print("─"*62)
    for i, row in bowl_lb.head(10).iterrows():
        print(f"  {i+1:2}. {row['player']:<25} ({row['team']:<35}) "
              f" Career: {int(row['career_wickets'])}  Proj: ~{row['projected_wickets_2026']}")

    print("\n🔮  KEY PREDICTIONS")
    print("─"*62)
    t1,b1,w1 = pred_df.iloc[0], bat_lb.iloc[0], bowl_lb.iloc[0]
    print(f"  Champion         → {t1['team']} ({t1['win_probability']:.1f}%)")
    print(f"  Top Run Scorer   → {b1['player']} ({b1['team']})  ~{b1['projected_runs_2026']} runs")
    print(f"  Top Wicket Taker → {w1['player']} ({w1['team']})  ~{w1['projected_wickets_2026']} wickets")

    print("\n📊  SQUAD STRENGTH RANKINGS")
    print("─"*62)
    ss = sorted(squad_strength.items(), key=lambda x: x[1]["combined"], reverse=True)
    for i,(team,v) in enumerate(ss):
        print(f"  {i+1:2}. {team:<35}  Bat:{v['bat']:7.1f}  Bowl:{v['bowl']:6.1f}  Combined:{v['combined']:7.1f}")

    # Save JSON
    results = {
        "model_cv_auc": float(cv_auc),
        "win_probabilities": pred_df[["team","win_probability"]].to_dict(orient="records"),
        "top_run_scorers": bat_lb.head(10)[["player","team","career_runs","projected_runs_2026"]].to_dict(orient="records"),
        "top_wicket_takers": bowl_lb.head(10)[["player","team","career_wickets","projected_wickets_2026"]].to_dict(orient="records"),
        "squad_strengths": [{"team":t,"bat":v["bat"],"bowl":v["bowl"],"combined":v["combined"]} for t,v in ss],
        "player_batting":  bat_lb.head(25)[["player","team","career_runs","w_runs","projected_runs_2026"]].to_dict(orient="records"),
        "player_bowling":  bowl_lb.head(25)[["player","team","career_wickets","w_wickets","projected_wickets_2026"]].to_dict(orient="records"),
    }
    with open("predictions_2026.json","w") as f:
        json.dump(results, f, indent=2)
    print("\n✅  Saved → predictions_2026.json")
    print("="*62)
    return results


if __name__ == "__main__":
    run()
