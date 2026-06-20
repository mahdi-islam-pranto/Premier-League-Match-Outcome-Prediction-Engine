"""
src/pipeline/predict_pipeline.py
----------------------------------
Given a home team, away team, and match date, builds the same 18
pre-match features that DataTransformation produced during training,
scales them with the saved preprocessor, and returns a prediction.

HOW IT WORKS
------------
At inference time we don't re-run the full feature-engineering loop.
Instead, DataTransformation saved a snapshot of every team's state
(Elo, last-5-match form, home/away splits, H2H records) at the end
of the training dataset into artifacts/team_states.json.

predict_pipeline.py loads that snapshot and uses it to build the
feature vector for the requested match — exactly the same 18 features
the model was trained on, in exactly the same order.

ARTIFACTS REQUIRED (all produced by the training pipeline):
  artifacts/model.pkl          — the tuned best classifier
  artifacts/preprocessor.pkl   — fitted StandardScaler pipeline
  artifacts/team_states.json   — team Elo + form snapshots

UNKNOWN TEAMS
-------------
If a team is not in team_states.json (newly promoted club), we fall back to:
  Elo     → 1500 (league average)
  Form    → all zeros (no history)
  H2H     → all zeros (no history)
  Rest    → 14 days (typical mid-season gap)
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, date
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.exception import CustomException
from src.logger    import logging
from src.utils     import load_object


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictPipelineConfig:
    model_path:        str   = os.path.join('artifacts', 'model.pkl')
    preprocessor_path: str   = os.path.join('artifacts', 'preprocessor.pkl')
    team_states_path:  str   = os.path.join('artifacts', 'team_states.json')
    default_elo:       float = 1500.0
    default_rest_days: int   = 14
    form_window:       int   = 5



LABEL_MAP    = {0: 'H',        1: 'D',    2: 'A'}
READABLE_MAP = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}

FEATURE_COLUMNS = [
    'home_elo', 'away_elo', 'elo_diff',
    'home_form_pts', 'away_form_pts',
    'home_form_gf',  'home_form_ga',
    'away_form_gf',  'away_form_ga',
    'home_home_pts', 'away_away_pts',
    'home_home_gf',  'away_away_gf',
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
    'home_days_rest', 'away_days_rest',
]



# MAIN PIPELINE CLASS


class PredictPipeline:
    def __init__(self):
        self.config        = PredictPipelineConfig()
        self._model        = None
        self._preprocessor = None
        self._team_states  = None

    # Lazy-load all artifacts on first call 
    def _load_artifacts(self):
        if self._model is not None:
            return

        for path, label in [
            (self.config.model_path,        'model.pkl'),
            (self.config.preprocessor_path, 'preprocessor.pkl'),
            (self.config.team_states_path,  'team_states.json'),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"'{label}' not found at '{path}'. "
                    "Run the full training pipeline first."
                )

        self._model        = load_object(self.config.model_path)
        self._preprocessor = load_object(self.config.preprocessor_path)

        with open(self.config.team_states_path, 'r') as f:
            self._team_states = json.load(f)

        logging.info(
            f"Artifacts loaded — "
            f"{len(self._team_states.get('teams', {}))} teams | "
            f"states generated: {self._team_states.get('generated_at')}"
        )

    # Fetch team state with safe fallback for unknown teams 
    def _get_team_state(self, team: str) -> Dict:
        state = self._team_states.get('teams', {}).get(team)
        if state is None:
            logging.warning(
                f"Team '{team}' not in team_states.json. "
                "Using league-average defaults. "
                "Check for spelling differences (e.g. 'Man City' vs 'Manchester City')."
            )
            return {
                'elo': self.config.default_elo,
                'last_date': None,
                'recent': [],
                'recent_home': [],
                'recent_away': [],
            }
        return state

    # Days since last match 
    def _days_rest(self, last_date_str: Optional[str], match_date: date) -> int:
        if not last_date_str:
            return self.config.default_rest_days
        try:
            last  = datetime.strptime(last_date_str, '%Y-%m-%d').date()
            delta = (match_date - last).days
            return max(delta, 1)
        except (ValueError, TypeError):
            return self.config.default_rest_days

    # now Build the 18-feature vector 
    def _build_features(self, home: str, away: str, match_date: date) -> Dict[str, float]:
        W = self.config.form_window

        hs = self._get_team_state(home)
        as_ = self._get_team_state(away)

        home_elo = hs['elo']
        away_elo = as_['elo']

        home_recent = hs.get('recent', [])[-W:]
        away_recent = as_.get('recent', [])[-W:]

        home_home_rec = hs.get('recent_home', [])[-W:]
        away_away_rec = as_.get('recent_away', [])[-W:]

        pair_key    = f"{min(home, away)}|||{max(home, away)}"
        h2h_records = self._team_states.get('h2h', {}).get(pair_key, [])[-W:]

        return {
            'home_elo':       round(home_elo, 2),
            'away_elo':       round(away_elo, 2),
            'elo_diff':       round(home_elo - away_elo, 2),
            'home_form_pts':  sum(m['pts'] for m in home_recent),
            'away_form_pts':  sum(m['pts'] for m in away_recent),
            'home_form_gf':   sum(m['gf']  for m in home_recent),
            'home_form_ga':   sum(m['ga']  for m in home_recent),
            'away_form_gf':   sum(m['gf']  for m in away_recent),
            'away_form_ga':   sum(m['ga']  for m in away_recent),
            'home_home_pts':  sum(m['pts'] for m in home_home_rec),
            'away_away_pts':  sum(m['pts'] for m in away_away_rec),
            'home_home_gf':   sum(m['gf']  for m in home_home_rec),
            'away_away_gf':   sum(m['gf']  for m in away_away_rec),
            'h2h_home_wins':  sum(1 for r in h2h_records if r == f'{home}_win'),
            'h2h_away_wins':  sum(1 for r in h2h_records if r == f'{away}_win'),
            'h2h_draws':      sum(1 for r in h2h_records if r == 'draw'),
            'home_days_rest': self._days_rest(hs.get('last_date'), match_date),
            'away_days_rest': self._days_rest(as_.get('last_date'), match_date),
        }




    # Main predict method
    def predict(self, home_team: str, away_team: str, match_date: str) -> Dict:
        """
        Args:
            home_team  : e.g. "Arsenal"
            away_team  : e.g. "Chelsea"
            match_date : "YYYY-MM-DD"

        Returns a dict ready to be serialised to JSON by FastAPI:
        {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "match_date": "2026-08-15",
            "predicted_result": "H",
            "predicted_label": "Home Win",
            "probabilities": {"home_win": 0.52, "draw": 0.24, "away_win": 0.24},
            "confidence": 0.52,
            "elo_ratings": {"home": 1623.4, "away": 1598.2, "diff": 25.2},
            "features_used": { ...all 18 raw feature values... }
        }
        """
        try:
            self._load_artifacts()

            match_dt = datetime.strptime(match_date, '%Y-%m-%d').date()
            features = self._build_features(home_team, away_team, match_dt)

            X_raw    = np.array([[features[col] for col in FEATURE_COLUMNS]])
            X_scaled = self._preprocessor.transform(X_raw)

            pred_class = int(self._model.predict(X_scaled)[0])
            proba      = self._model.predict_proba(X_scaled)[0]

            result = {
                'home_team':        home_team,
                'away_team':        away_team,
                'match_date':       match_date,
                'predicted_result': LABEL_MAP[pred_class],
                'predicted_label':  READABLE_MAP[pred_class],
                'probabilities': {
                    'home_win': round(float(proba[0]), 4),
                    'draw':     round(float(proba[1]), 4),
                    'away_win': round(float(proba[2]), 4),
                },
                'confidence':  round(float(proba.max()), 4),
                'elo_ratings': {
                    'home': features['home_elo'],
                    'away': features['away_elo'],
                    'diff': features['elo_diff'],
                },
                'features_used': features,
            }

            logging.info(
                f"{home_team} vs {away_team} → "
                f"{result['predicted_label']} "
                f"(confidence {result['confidence']:.1%})"
            )
            return result

        except Exception as e:
            raise CustomException(e, sys)

    def available_teams(self) -> List[str]:
        """Returns sorted list of all teams in team_states.json."""
        self._load_artifacts()
        return sorted(self._team_states.get('teams', {}).keys())
    
    
# test the pipeline
if __name__ == "__main__":
    tottenham_arsenal = PredictPipeline().predict(
        home_team="Arsenal",
        away_team="Tottenham",
        match_date="2026-07-01",
    )
    
    print(tottenham_arsenal)
    
    logging.info(f'Prediction Results:',
                 tottenham_arsenal)