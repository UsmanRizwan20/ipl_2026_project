# IPL 2026 Prediction Engine
### ML Project | 18 Years of IPL Data (2008–2025)

Predicts:
- 🏆 Tournament win probability for each team
- 🏏 Top run scorer for IPL 2026
- 🎳 Top wicket taker for IPL 2026

---

## Setup

### 1. Place your data files
Put all 4 data files inside the `data/` folder:
```
data/
  ball_by_ball_data.csv        ← extract from ball_by_ball_data.zip first
  ipl_matches_data.csv
  players-data-updated.csv
  teams_data.csv
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the ML predictor
```bash
python ipl_2026_predictor.py
```
Outputs predictions to console + saves `predictions_2026.json`

### 4. Launch the interactive dashboard
```bash
python dashboard.py
```
Opens a Plotly Dash app at http://127.0.0.1:8050

---

## Project Structure
```
ipl_2026_project/
├── ipl_2026_predictor.py   ← ML model (run this first)
├── dashboard.py            ← Interactive web dashboard
├── requirements.txt
├── README.md
└── data/                   ← Put your CSV files here
```

## Model Details
- **Algorithm**: Gradient Boosting Classifier (sklearn)
- **CV AUC**: ~0.87
- **Features**: Win rate, toss win rate, historical titles, 
                playoff appearances, squad batting/bowling strength,
                recency-weighted player stats (exponential decay, half-life=3 seasons)
- **Prediction blend**: Squad strength (40%) + Model (25%) + History (20%) + Recent form (15%)
