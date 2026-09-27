# **KickOffIQ:** Premier League Match Outcome Prediction Engine

**Live Demo:** [http://138.252.115.100:8003/](http://138.252.115.100:8003/)

## Overview

An end-to-end machine learning project that predicts the result of an English Premier League football match — **Home Win, Draw, or Away Win** — using only information available before the match starts.

Built with: Python · Scikit-learn · XGBoost · LightGBM · MLflow · FastAPI · Streamlit

It includes:

- a full training pipeline for preprocessing, feature engineering, and model selection
- MLflow experiment tracking for training and tuning
- a FastAPI service for serving predictions
- a Streamlit frontend for interactive match prediction and league analytics

> Given two Premier League teams, a home/away assignment, and a match date, predict whether the match will end in a Home Win, Draw, or Away Win — using only statistics from matches that have already been played.

This is a **3-class classification problem**. The model predicts one of three match outcomes:

| Code  | Meaning  | ~Frequency |
| ----- | -------- | ---------- |
| `H` | Home Win | 43%        |
| `D` | Draw     | 27%        |
| `A` | Away Win | 30%        |

---

## Dataset

- **Source:** Football-Data.co.uk (10 EPL seasons, 2016–17 to 2025–26)
- **Size:** ~3,800 matches (380 matches × 10 seasons)
- **Teams:** 34 unique clubs (teams are promoted and relegated each season)
- **Raw columns:** Date, HomeTeam, AwayTeam, Goals (FT + HT), Result (FT + HT), Referee, Shots, Shots on Target, Fouls, Corners, Yellow Cards, Red Cards

## The 18 Engineered Features

All 18 features are computed per-match, per-team, using only prior matches. None require any in-match data.

### Group 1: Elo Ratings (3 features)

**What is Elo?** Elo is a rating system originally invented for chess. Every team starts at 1500. After each match:

- The winner gains points, the loser loses points
- The amount transferred depends on how surprising the result was
- Beating a much stronger team gains more points than beating a weaker one

**Formula:**

```
Expected score for home team:  E_home = 1 / (1 + 10^((Elo_away - Elo_home) / 400))
New Elo for home team:         Elo_home_new = Elo_home + K × (actual_score - E_home)
  where actual_score = 1.0 (win), 0.5 (draw), 0.0 (loss)
  and K = 20 (how fast ratings change)
```

**The 3 Elo features:**

| Feature      | What it captures                                                                                |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `home_elo` | Home team's current strength rating                                                             |
| `away_elo` | Away team's current strength rating                                                             |
| `elo_diff` | `home_elo - away_elo` — the single most predictive feature. Positive = home team is stronger |

After 10 seasons, top clubs like Arsenal/Man City sit around 1600–1650. Relegated clubs hover around 1350–1400. A difference of +100 Elo points means a significant advantage.

### Group 2: Overall Rolling Form — Last 5 Matches (6 features)

For each team, look at their last 5 matches (regardless of home/away venue) and compute:

| Feature           | What it captures                                                                     |
| ----------------- | ------------------------------------------------------------------------------------ |
| `home_form_pts` | Points (3=win, 1=draw, 0=loss) in last 5 matches. Max = 15. Captures recent momentum |
| `away_form_pts` | Same for the away team                                                               |
| `home_form_gf`  | Goals scored in last 5 matches. Captures attacking form                              |
| `home_form_ga`  | Goals conceded in last 5 matches. Captures defensive form                            |
| `away_form_gf`  | Same for away team                                                                   |
| `away_form_ga`  | Same for away team                                                                   |

**Why last 5 and not last 10?** 5 is the standard "form window" in football analytics — short enough to capture current form, long enough to smooth out a single lucky/unlucky result.

### Group 3: Venue-Specific Form — Last 5 Home/Away Games (4 features)

Some teams are very strong at home but poor travelers. This group captures that pattern:

| Feature           | What it captures                                            |
| ----------------- | ----------------------------------------------------------- |
| `home_home_pts` | Home team's points in their last 5**home** games only |
| `home_home_gf`  | Home team's goals scored in their last 5 home games         |
| `away_away_pts` | Away team's points in their last 5**away** games only |
| `away_away_gf`  | Away team's goals scored in their last 5 away games         |

**Example:** A team might have `home_form_pts = 12` (great overall) but `away_away_pts = 3` (poor away form). The overall form hides the venue split.

### Group 4: Head-to-Head Record (3 features)

The last 5 meetings between these two specific teams:

| Feature           | What it captures                                       |
| ----------------- | ------------------------------------------------------ |
| `h2h_home_wins` | Times the current home team won in last 5 H2H meetings |
| `h2h_away_wins` | Times the current away team won in last 5 H2H meetings |
| `h2h_draws`     | Draws in last 5 H2H meetings                           |

**Why H2H?** Some teams have psychological edges over specific opponents that persist even when their Elo ratings are similar. Classic example: certain teams consistently outperform their Elo predictions against specific rivals.

### Group 5: Rest Days (2 features)

| Feature            | What it captures                     |
| ------------------ | ------------------------------------ |
| `home_days_rest` | Days since the home team last played |
| `away_days_rest` | Days since the away team last played |

**Why rest days?** A team playing their third game in 7 days (common in cup competitions and congested schedules) is at a meaningful disadvantage — tired legs, higher injury risk, possible rotation. A team with 14 days rest is fresher.

Default value for the very first match of a team (no history): 30 days.

## Key Components

- `src/components/data_ingestion.py` — loads the raw EPL dataset, splits it by season, and writes chronological train/val/test CSV files to `artifacts/`
- `src/components/data_transformation.py` — engineers pre-match features, builds the scaler, saves artifacts, and writes feature arrays
- `src/components/model_trainer.py` — compares classifiers, tunes the best model with `RandomizedSearchCV`, saves the final model, and creates a JSON report
- `src/pipeline/prediction_pipeline.py` — loads trained artifacts and builds the exact 18-feature inference vector for new matches
- `app.py` — FastAPI application exposing prediction and metadata endpoints
- `frontend.py` — Streamlit dashboard for single-match prediction, batch fixtures, and Elo rankings

---

## Repository Structure

```text
epl-predictor/
├── artifacts/                    # All trained model artifacts (git-ignored)
│   ├── model.pkl                 # Best trained classifier
│   ├── preprocessor.pkl          # Fitted StandardScaler
│   ├── team_states.json          # Team Elo + form snapshots for inference
│   ├── train.csv / val.csv / test.csv
│   ├── raw.csv
│   ├── featured_data.csv         # Full dataset with 18 engineered features
│   └── model_report.json         # All model metrics from training run
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     # Chronological train/val/test split
│   │   ├── data_transformation.py # Feature engineering + preprocessing
│   │   └── model_trainer.py      # Model comparison + tuning + MLflow
│   ├── pipeline/
│   │   └── predict_pipeline.py   # Loads artifacts, builds features, predicts
│   ├── exception.py              # Custom exception with file + line info
│   ├── logger.py                 # Timestamped file + console logging
│   └── utils.py                  # save_object / load_object helpers
│
├── notebook/
│   ├── data/                     # Raw season CSV files
│   └── EDA.ipynb                 # Exploratory data analysis
│
├── app.py                        # FastAPI application (all endpoints)
├── streamlit_app.py              # Streamlit frontend (3 pages)
├── mlflow.db                     # MLflow experiment tracking database
├── requirements.txt
└── README.md
```

---

## Setup

1. Open a terminal in the repository root:

```bash
cd "D:\projects\Premier League Match Prediction"
```

2. Create and activate a Python virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> If you are using a different environment manager, make sure the current interpreter points to the repo root.

---

## Training the Model

Run the full training pipeline from the project root:

```bash
python -m src.components.model_trainer
```

This script will:

- ingest the dataset from `notebooks/datasets/pl-matches-dataset-16-26.csv`
- engineer features and scale them
- compare multiple classifiers using weighted F1
- tune the best model
- save the final model and report under `artifacts/`

### Generated artifacts

- `artifacts/model.pkl`
- `artifacts/model_report.json`
- `artifacts/preprocessor.pkl`
- `artifacts/team_states.json`
- `artifacts/train_array.npy`, `val_array.npy`, `test_array.npy`

---

## MLflow Tracking

The pipeline uses MLflow with a local SQLite backend (`mlflow.db`).
Launch the MLflow UI from the project root:

```bash
mlflow ui --port 5001
```

Then open:

- `http://127.0.0.1:5001`

This interface shows:

- phase 1 model comparisons
- phase 2 hyperparameter tuning
- model metrics and artifacts

### Screenshots

![MLflow run overview](<screenshots/mlflow%20epl%20evaluation%20run%20overview.png>)

![MLflow run matrices overview](<screenshots/mlflow%20epl%20evaluation%20run%20matrices%20overview.png>)

---

## FastAPI Service

Start the API server from the project root:

```bash
uvicorn app:app --reload --port 8000
```

Then visit:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Main endpoints

- `GET /health` — health check + artifact status
- `GET /teams` — list of supported team names
- `POST /predict` — single match prediction
- `POST /predict/batch` — batch predictions for up to 10 matches

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "match_date": "2026-08-16"
  }'
```

### Batch example

```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "matches": [
      {"home_team": "Arsenal", "away_team": "Chelsea", "match_date": "2026-08-16"},
      {"home_team": "Liverpool", "away_team": "Man City", "match_date": "2026-08-16"}
    ]
  }'
```

![FastAPI endpoints](<screenshots/fastapi%20endpoints%20epl.png>)

---

## Streamlit Frontend

Start the frontend from the project root:

```bash
streamlit run frontend.py
```

The app connects to the FastAPI backend at `http://127.0.0.1:8000`.

### Features

- Single-match prediction card
- Batch fixture prediction display
- Elo power rankings and league analytics

### Screenshots

![Streamlit match prediction](<screenshots/epl%20streamlit%20match%20prediction.png>)

![Streamlit elo power ranking](<screenshots/epl%20streamlit%20elo%20power%20ranking.png>)

![Streamlit batch matches predictor](<screenshots/epl%20streamlit%20batch%20matches%20predictor.png>)

---

## Notes

- The model uses only pre-match features, so it does not depend on in-game statistics like shots or cards.
- Unknown teams are handled gracefully by falling back to average Elo and default form values.
- The dataset source is expected at `notebooks/datasets/pl-matches-dataset-16-26.csv`.

## Troubleshooting

- If you see `ModuleNotFoundError: No module named 'src'`, make sure you are running commands from the repository root.
- Use `python -m src.components.model_trainer` and `python -m src.pipeline.prediction_pipeline` instead of running module files from nested folders.

## License

This project is configured as a personal codebase. Feel free to reuse the structure and ideas for your own EPL prediction work.
