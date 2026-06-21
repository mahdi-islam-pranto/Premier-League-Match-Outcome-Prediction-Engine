# ----------- data transformation ------------------
"""Transforms raw EPL match data into a leakage-free, model-ready feature matrix.

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
  full_time_result: H → 0, D → 1, A → 2"""
  
  

# ----------------------- model trainer --------------------------
"""
Trains multiple classifiers on the pre-match EPL feature matrix,
selects the best by validation F1 (weighted), tunes hyperparameters,
and produces a full evaluation report.

Used WEIGHTED F1 matric instead of ACCURACY.
-----------------------
Accuracy is misleading for this problem. The class distribution is roughly:
  Home Win  ~43%  | Draw ~27%  | Away Win ~30%
A naive model that always predicts "Home Win" gets 43% accuracy for free.
We use WEIGHTED F1 as the primary selection metric so that the Draw class
(the hardest and most minority) is not silently ignored. Per-class F1 is
also reported so you can see exactly how well the model handles each outcome.

MODELS COMPARED (Phase 1 — default hyperparameters):
  1. Logistic Regression     — linear baseline, interpretable, fast
  2. Random Forest           — robust ensemble, handles non-linearities
  3. XGBoost                 — usually best on tabular sports data
  4. LightGBM                — faster XGBoost alternative, good on small data
  5. K-Nearest Neighbours    — distance-based, uses the scaled Elo/form features

PHASE 2 — Hyperparameter tuning on the winner via RandomizedSearchCV
  (GridSearch is too slow; RandomizedSearch gives 90% of the benefit in 10% of the time)
  
  
  MLflow is integrated at every stage so every experiment run is fully
reproducible and inspectable from the MLflow UI.
 
HOW TO VIEW RESULTS:
  After running this file, launch the MLflow UI from your project root:
    $ mlflow ui --port 5001
  Then open: http://127.0.0.1:5001
 
  You will see one parent run per training session, with nested child
  runs for each model compared in Phase 1 and the tuned winner in Phase 2.
 

ARTIFACTS SAVED:
  artifacts/model.pkl          — best trained model (after tuning)
  artifacts/model_report.json  — full metrics for all models (for README / logging)
"""


# ----------------- prediction pipeline ----------------------------
"""
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