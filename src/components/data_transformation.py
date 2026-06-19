"""
Transforms raw EPL match data into a leakage-free, model-ready feature matrix.

Here, we must first ENGINEER features that didn't exist
in the raw data — because all the raw match stats (shots, fouls, corners, cards)
are in-match information your model won't have access to before kickoff.

The engineered features are all computed using ONLY matches that happened BEFORE
the current match for each team. This is called "point-in-time correctness"
and is what separates a real ML project from a leaky one.

FEATURES ENGINEERED (all pre-match, zero leakage):
  - Elo rating (home & away)      : rolling team strength estimate
  - Elo difference                : single strongest pre-match signal
  - Rolling form pts (last 5)     : recent form (any venue)
  - Rolling goals scored/conceded : attacking/defensive form (last 5)
  - Home-specific form            : last 5 home games for home team
  - Away-specific form            : last 5 away games for away team
  - Head-to-head record           : last 5 meetings between these two teams
  - Days rest                     : fatigue/recovery proxy

COLUMNS DROPPED (in-match / post-match / leaky):
  - half_time_* (known only at HT)
  - home/away team shots, shots on target, fouls, corners, cards
  - full_time_home_goals, full_time_away_goals  (derived from target)
  - home_points, away_points                    (direct encoding of target)
  - match_referee                               (weak signal, hard at inference)
  - match_date, season                          (used for feature eng then dropped)

TARGET:
  full_time_result: H → 0, D → 1, A → 2
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict, deque

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    train_arr_path: str             = os.path.join('artifacts', 'train_array.npy')
    val_arr_path: str               = os.path.join('artifacts', 'val_array.npy')
    test_arr_path: str              = os.path.join('artifacts', 'test_array.npy')
    featured_data_path: str         = os.path.join('artifacts', 'featured_data.csv')

    # Rolling window for form features
    FORM_WINDOW: int = 5

    # Elo hyperparameters
    ELO_START: float  = 1500.0
    ELO_K:     float  = 20.0


# TARGET MAPPING # Home Win=0, Draw=1, Away Win=2
RESULT_MAP = {'H': 0, 'D': 1, 'A': 2}   


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    # points earned from a result
    def _points(self, result: str, perspective: str) -> int:
        """
        Returns points (3/1/0) for a team given the full_time_result string
        and whether this team was the 'home' or 'away' side.
        """
        if result == 'D':
            return 1
        if (result == 'H' and perspective == 'home') or \
           (result == 'A' and perspective == 'away'):
            return 3
        return 0

    # engineer all pre-match features 
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates through every match in chronological order.
        For each match, looks up each team's historical stats (only prior matches)
        and builds the feature row BEFORE updating the history with this match.

        This guarantees zero temporal leakage.
        """
        logging.info("Engineering pre-match features (Elo, form, H2H, rest days)")

        W = self.config.FORM_WINDOW

        # Per-team running histories
        elo          = defaultdict(lambda: self.config.ELO_START)
        last_date    = {}                        # team → last match date
        # deques hold dicts of {'pts', 'gf', 'ga', 'venue'}
        team_history = defaultdict(lambda: deque(maxlen=50))

        # ── Per-pair H2H history ──────────────────────────────────────────────
        # key = frozenset({team_a, team_b}) → deque of 'H_team_won' or 'A_team_won' or 'D'
        h2h_history  = defaultdict(lambda: deque(maxlen=W))

        rows = []

        
        for _, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            date = row['match_date']
            ftr  = row['full_time_result']
            hg   = row['full_time_home_goals']
            ag   = row['full_time_away_goals']

            # SNAPSHOT: read features BEFORE updating with this match

            # 1. Elo ratings
            home_elo = elo[home]
            away_elo = elo[away]
            elo_diff = home_elo - away_elo

            # 2. Overall rolling form (last W matches, any venue)
            home_hist = list(team_history[home])[-W:]
            away_hist = list(team_history[away])[-W:]

            home_form_pts = sum(m['pts'] for m in home_hist)
            away_form_pts = sum(m['pts'] for m in away_hist)
            home_form_gf  = sum(m['gf']  for m in home_hist)
            home_form_ga  = sum(m['ga']  for m in home_hist)
            away_form_gf  = sum(m['gf']  for m in away_hist)
            away_form_ga  = sum(m['ga']  for m in away_hist)

            # 3. Venue-specific form (home team's last W home games,
            #                         away team's last W away games)
            home_home_hist = [m for m in team_history[home] if m['venue'] == 'home'][-W:]
            away_away_hist = [m for m in team_history[away] if m['venue'] == 'away'][-W:]

            home_home_pts  = sum(m['pts'] for m in home_home_hist)
            away_away_pts  = sum(m['pts'] for m in away_away_hist)
            home_home_gf   = sum(m['gf']  for m in home_home_hist)
            away_away_gf   = sum(m['gf']  for m in away_away_hist)

            # 4. Head-to-head (last W meetings, any venue)
            pair_key  = frozenset({home, away})
            h2h       = list(h2h_history[pair_key])
            h2h_home  = sum(1 for r in h2h if r == f'{home}_win')
            h2h_away  = sum(1 for r in h2h if r == f'{away}_win')
            h2h_draw  = sum(1 for r in h2h if r == 'draw')

            # 5. Days since last match (rest / fatigue)
            home_rest = (date - last_date[home]).days if home in last_date else 30
            away_rest = (date - last_date[away]).days if away in last_date else 30

            rows.append({
                # identifiers (will be dropped before training)
                'match_date':       date,
                'home_team':        home,
                'away_team':        away,
                'season':           row.get('season', None),

                # PRE-MATCH FEATURES (model inputs)
                'home_elo':         round(home_elo, 2),
                'away_elo':         round(away_elo, 2),
                'elo_diff':         round(elo_diff, 2),

                'home_form_pts':    home_form_pts,
                'away_form_pts':    away_form_pts,
                'home_form_gf':     home_form_gf,
                'home_form_ga':     home_form_ga,
                'away_form_gf':     away_form_gf,
                'away_form_ga':     away_form_ga,

                'home_home_pts':    home_home_pts,   # home team's home form
                'away_away_pts':    away_away_pts,   # away team's away form
                'home_home_gf':     home_home_gf,
                'away_away_gf':     away_away_gf,

                'h2h_home_wins':    h2h_home,
                'h2h_away_wins':    h2h_away,
                'h2h_draws':        h2h_draw,

                'home_days_rest':   home_rest,
                'away_days_rest':   away_rest,

                # TARGET
                'target':           RESULT_MAP[ftr],
            })

            # UPDATE histories AFTER snapshotting 

            # Elo update
            exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            exp_away = 1 - exp_home
            score_home = 1.0 if ftr == 'H' else (0.5 if ftr == 'D' else 0.0)
            score_away = 1.0 - score_home
            elo[home] = home_elo + self.config.ELO_K * (score_home - exp_home)
            elo[away] = away_elo + self.config.ELO_K * (score_away - exp_away)

            # Team form history
            team_history[home].append({
                'pts':   self._points(ftr, 'home'),
                'gf':    hg,
                'ga':    ag,
                'venue': 'home',
            })
            team_history[away].append({
                'pts':   self._points(ftr, 'away'),
                'gf':    ag,
                'ga':    hg,
                'venue': 'away',
            })

            # H2H history
            if ftr == 'H':
                h2h_history[pair_key].append(f'{home}_win')
            elif ftr == 'A':
                h2h_history[pair_key].append(f'{away}_win')
            else:
                h2h_history[pair_key].append('draw')

            # Last-match date
            last_date[home] = date
            last_date[away] = date

        return pd.DataFrame(rows)

    # Build sklearn preprocessing pipeline 
    def get_preprocessor(self) -> Pipeline:
        """
        Builds a simple numerical pipeline:
          1. SimpleImputer (median) — handles any early-season NaN-equivalent rows
          2. StandardScaler         — normalises for distance-based models (LR, SVM)
                                      harmless for tree-based (RF, XGBoost)

        There are NO categorical columns fed to the model:
          - Team identity is captured by Elo (continuous strength rating)
          - No referee or venue string is included
        """
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
        ])
        return num_pipeline

    # Final feature columns 
    @property
    def feature_columns(self):
        return [
            'home_elo', 'away_elo', 'elo_diff',
            'home_form_pts', 'away_form_pts',
            'home_form_gf', 'home_form_ga',
            'away_form_gf', 'away_form_ga',
            'home_home_pts', 'away_away_pts',
            'home_home_gf', 'away_away_gf',
            'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'home_days_rest', 'away_days_rest',
        ]

    # Main entry point for transforming the data
    def initiate_data_transformation(
        self,
        train_path: str,
        val_path:   str,
        test_path:  str,
    ):
        """
        Reads the three chronological splits produced by DataIngestion,
        concatenates them back into a single sorted DataFrame (so that rolling
        features computed at the start of val/test have the full prior history),
        engineers features, re-splits, applies sklearn preprocessing, and saves:

          artifacts/preprocessor.pkl
          artifacts/train_array.npy
          artifacts/val_array.npy
          artifacts/test_array.npy
          artifacts/featured_data.csv

        Returns:
            (train_array, val_array, test_array, preprocessor_path)
        """
        try:
            logging.info("Reading train / val / test splits from DataIngestion")
            train_df = pd.read_csv(train_path)
            val_df   = pd.read_csv(val_path)
            test_df  = pd.read_csv(test_path)

            # recombine before feature engineering
            # We must compute rolling stats on the full chronological sequence.
            # If we engineered features per-split, the first matches in val/test
            # would have empty look-back windows (their history is in train_df).
            full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            full_df['match_date'] = pd.to_datetime(full_df['match_date'])
            full_df = full_df.sort_values('match_date').reset_index(drop=True)

            logging.info(f"Full dataset shape before feature engineering: {full_df.shape}")

            # Engineer pre-match features 
            featured_df = self._engineer_features(full_df)
            logging.info(f"Featured dataset shape: {featured_df.shape}")

            # Save for notebook inspection
            os.makedirs(os.path.dirname(self.config.featured_data_path), exist_ok=True)
            featured_df.to_csv(self.config.featured_data_path, index=False)
            logging.info(f"Saved featured data → {self.config.featured_data_path}")

            # Re-split using season boundaries (same logic as DataIngestion) 
            seasons_sorted = sorted(featured_df['season'].dropna().unique())
            train_seasons  = seasons_sorted[:-2]
            val_season     = seasons_sorted[-2]
            test_season    = seasons_sorted[-1]

            train_feat = featured_df[featured_df['season'].isin(train_seasons)]
            val_feat   = featured_df[featured_df['season'] == val_season]
            test_feat  = featured_df[featured_df['season'] == test_season]

            logging.info(
                f"Re-split sizes — Train: {len(train_feat)} | "
                f"Val: {len(val_feat)} | Test: {len(test_feat)}"
            )

            # Separate features and target 
            X_train = train_feat[self.feature_columns]
            y_train = train_feat['target'].values

            X_val   = val_feat[self.feature_columns]
            y_val   = val_feat['target'].values

            X_test  = test_feat[self.feature_columns]
            y_test  = test_feat['target'].values

            # Fit preprocessor on train ONLY, transform all three splits
            logging.info("Fitting preprocessor on training data")
            preprocessor = self.get_preprocessor()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_val_scaled   = preprocessor.transform(X_val)
            X_test_scaled  = preprocessor.transform(X_test)

            # Build final arrays (features + target as last column)
            train_array = np.c_[X_train_scaled, y_train]
            val_array   = np.c_[X_val_scaled,   y_val]
            test_array  = np.c_[X_test_scaled,  y_test]

            # ── Save everything ───────────────────────────────────────────────
            logging.info("Saving preprocessor and numpy arrays")
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            np.save(self.config.train_arr_path, train_array)
            np.save(self.config.val_arr_path,   val_array)
            np.save(self.config.test_arr_path,  test_array)

            logging.info(f"Preprocessor saved → {self.config.preprocessor_obj_file_path}")
            logging.info(f"train_array shape: {train_array.shape}")
            logging.info(f"val_array shape:   {val_array.shape}")
            logging.info(f"test_array shape:  {test_array.shape}")

            return (
                train_array,
                val_array,
                test_array,
                self.config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)



# STANDALONE RUN (triggers full pipeline)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, val_arr, test_arr, preprocessor_path = \
        transformation.initiate_data_transformation(train_path, val_path, test_path)

    print("\n✅ DataTransformation complete")
    print(f"   train_array : {train_arr.shape}")
    print(f"   val_array   : {val_arr.shape}")
    print(f"   test_array  : {test_arr.shape}")
    print(f"   preprocessor: {preprocessor_path}")
    print(f"\n   Features ({len(transformation.feature_columns)}):")
    for f in transformation.feature_columns:
        print(f"     - {f}")
    print(f"\n   Target encoding: H=0, D=1, A=2")