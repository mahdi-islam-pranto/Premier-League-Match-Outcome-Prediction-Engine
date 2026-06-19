"""
src/components/model_trainer.py
---------------------------------
Trains multiple classifiers on the pre-match EPL feature matrix,
selects the best by validation F1 (weighted), tunes hyperparameters,
and produces a full evaluation report.

WHY NOT JUST ACCURACY?
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

ARTIFACTS SAVED:
  artifacts/model.pkl          — best trained model (after tuning)
  artifacts/model_report.json  — full metrics for all models (for README / logging)
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, log_loss, classification_report,confusion_matrix
)

from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# CONFIG
@dataclass
class ModelTrainerConfig:
    model_file_path:  str = os.path.join('artifacts', 'model.pkl')
    report_file_path: str = os.path.join('artifacts', 'model_report.json')

    # Primary metric used for model selection and tuning
    # 'f1_weighted' balances all three classes by their support
    scoring_metric: str = 'f1_weighted'

    # RandomizedSearchCV settings
    n_iter:     int = 40
    cv_folds:   int = 5
    random_state: int = 42


# TARGET CLASS NAMES (for readable reports)

CLASS_NAMES = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
CLASSES     = [0, 1, 2]


# MAIN CLASS

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    # Define candidate models
    def _get_models(self) -> Dict[str, Any]:
        """
        Returns dict of {model_name: unfitted_estimator}.
        All models use class_weight='balanced' where supported
        so that the minority Draw class is not systematically ignored.
        """
        return {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.config.random_state,
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=self.config.random_state,
                n_jobs=-1,
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=self.config.random_state,
                verbosity=0,
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                class_weight='balanced',
                random_state=self.config.random_state,
                verbosity=-1,
                n_jobs=-1,
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=15,
                weights='distance',
                n_jobs=-1,
            ),
        }

    # Hyperparameter search spaces
    def _get_param_grids(self) -> Dict[str, Dict]:
        """
        Search spaces for RandomizedSearchCV on each model type.
        Focused on the most impactful parameters for each algorithm.
        """
        return {
            'Logistic Regression': {
                'C':       [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'solver':  ['lbfgs', 'saga'],
                'penalty': ['l2'],
            },
            'Random Forest': {
                'n_estimators':      [100, 200, 300, 500],
                'max_depth':         [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf':  [1, 2, 4],
                'max_features':      ['sqrt', 'log2', 0.5],
            },
            'XGBoost': {
                'n_estimators':    [100, 200, 300, 500],
                'learning_rate':   [0.01, 0.03, 0.05, 0.1, 0.2],
                'max_depth':       [3, 4, 5, 6, 7],
                'subsample':       [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1.0],
                'gamma':           [0, 0.1, 0.2, 0.5],
                'reg_alpha':       [0, 0.1, 0.5, 1.0],
                'reg_lambda':      [0.5, 1.0, 2.0, 5.0],
            },
            'LightGBM': {
                'n_estimators':  [100, 200, 300, 500],
                'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
                'num_leaves':    [15, 31, 63, 127],
                'max_depth':     [-1, 5, 10, 15],
                'min_child_samples': [10, 20, 30, 50],
                'subsample':     [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha':     [0, 0.1, 1.0],
                'reg_lambda':    [0, 0.1, 1.0],
            },
            'KNN': {
                'n_neighbors': [5, 7, 11, 15, 21, 31],
                'weights':     ['uniform', 'distance'],
                'metric':      ['euclidean', 'manhattan', 'minkowski'],
                'p':           [1, 2],
            },
        }

    # Evaluate one model on a split
    def _evaluate(
        self, model, X: np.ndarray, y: np.ndarray, split_name: str
    ) -> Dict:
        """Returns a metrics dict for a fitted model on X/y."""
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

        metrics = {
            'accuracy':    round(accuracy_score(y, y_pred), 4),
            'f1_weighted': round(f1_score(y, y_pred, average='weighted', zero_division=0), 4),
            'f1_per_class': {
                CLASS_NAMES[c]: round(
                    f1_score(y, y_pred, labels=[c], average='macro', zero_division=0), 4
                )
                for c in CLASSES
            },
            'log_loss': round(log_loss(y, y_prob), 4) if y_prob is not None else None,
        }
        logging.info(
            f"  [{split_name}] acc={metrics['accuracy']:.4f} | "
            f"f1_w={metrics['f1_weighted']:.4f} | "
            f"logloss={metrics['log_loss']}"
        )
        return metrics

    # Pretty-print confusion matrix
    def _log_confusion_matrix(self, model, X: np.ndarray, y: np.ndarray):
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred, labels=CLASSES)
        header = f"{'':12} | {'Pred H':>8} | {'Pred D':>8} | {'Pred A':>8}"
        logging.info("Confusion Matrix (rows=Actual, cols=Predicted):")
        logging.info(header)
        for i, row in enumerate(cm):
            logging.info(
                f"  {'Act '+CLASS_NAMES[i]:12} | {row[0]:>8} | {row[1]:>8} | {row[2]:>8}"
            )

    # Phase 1: compare all models on default params
    def _phase1_compare(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ) -> Tuple[str, Any, Dict]:
        """
        Fits every candidate model on training data, evaluates on val,
        returns the name and estimator of the winner.
        """
        models   = self._get_models()
        results  = {}
        best_name, best_model, best_score = None, None, -1.0

        logging.info("\n" + "="*60)
        logging.info("PHASE 1 — Default-parameter comparison")
        logging.info("="*60)

        for name, model in models.items():
            logging.info(f"\n→ Training: {name}")
            model.fit(X_train, y_train)

            train_metrics = self._evaluate(model, X_train, y_train, 'Train')
            val_metrics   = self._evaluate(model, X_val,   y_val,   'Val  ')

            results[name] = {
                'train': train_metrics,
                'val':   val_metrics,
            }

            val_f1 = val_metrics['f1_weighted']
            if val_f1 > best_score:
                best_score = val_f1
                best_name  = name
                best_model = model

        logging.info("\n" + "="*60)
        logging.info(f"Phase 1 winner: {best_name}  (val f1_weighted={best_score:.4f})")
        logging.info("="*60)
        return best_name, best_model, results

    # Phase 2: tune the winning model
    def _phase2_tune(
        self,
        best_name:  str,
        best_model: Any,
        X_train:    np.ndarray,
        y_train:    np.ndarray,
        X_val:      np.ndarray,
        y_val:      np.ndarray,
    ) -> Tuple[Any, Dict]:
        """
        Runs RandomizedSearchCV on the winning model from Phase 1.
        Uses StratifiedKFold so each fold keeps class proportions intact.
        Returns the tuned estimator and its metrics dict.
        """
        logging.info("\n" + "="*60)
        logging.info(f"PHASE 2 — Hyperparameter tuning: {best_name}")
        logging.info("="*60)

        param_grid = self._get_param_grids().get(best_name, {})
        if not param_grid:
            logging.info("No param grid defined for this model — skipping tuning")
            return best_model, {}

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        search = RandomizedSearchCV(
            estimator=best_model,
            param_distributions=param_grid,
            n_iter=self.config.n_iter,
            scoring=self.config.scoring_metric,
            cv=cv,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        # Tune on train only — val is untouched
        search.fit(X_train, y_train)

        tuned_model = search.best_estimator_
        logging.info(f"Best params found:\n  {search.best_params_}")
        logging.info(f"Best CV {self.config.scoring_metric}: {search.best_score_:.4f}")

        # Evaluate tuned model
        logging.info("\n--- Tuned model metrics ---")
        train_metrics = self._evaluate(tuned_model, X_train, y_train, 'Train')
        val_metrics   = self._evaluate(tuned_model, X_val,   y_val,   'Val  ')
        self._log_confusion_matrix(tuned_model, X_val, y_val)

        tuning_result = {
            'best_params':  search.best_params_,
            'cv_f1':        round(search.best_score_, 4),
            'train':        train_metrics,
            'val':          val_metrics,
        }
        return tuned_model, tuning_result




    # Main entry point 
    def initiate_model_trainer(
        self,
        train_array: np.ndarray,
        val_array:   np.ndarray,
    ) -> Tuple[float, str]:
        """
        Args:
            train_array : np.ndarray of shape (n_train, n_features + 1)
                          last column is the encoded target (0=H, 1=D, 2=A)
            val_array   : same format for the validation season

        Returns:
            (best_val_f1_weighted, model_file_path)

        Side effects:
            Saves artifacts/model.pkl and artifacts/model_report.json
        """
        try:
            # Unpack arrays 
            X_train, y_train = train_array[:, :-1], train_array[:, -1].astype(int)
            X_val,   y_val   = val_array[:,   :-1], val_array[:,   -1].astype(int)

            logging.info(f"X_train: {X_train.shape} | X_val: {X_val.shape}")
            logging.info(
                f"Class dist (train) — "
                f"H:{(y_train==0).sum()} D:{(y_train==1).sum()} A:{(y_train==2).sum()}"
            )
            logging.info(
                f"Class dist (val)   — "
                f"H:{(y_val==0).sum()} D:{(y_val==1).sum()} A:{(y_val==2).sum()}"
            )

            # Phase 1: compare all models 
            best_name, best_model, phase1_results = self._phase1_compare(
                X_train, y_train, X_val, y_val
            )

            # Phase 2: tune the winner 
            tuned_model, tuning_result = self._phase2_tune(
                best_name, best_model, X_train, y_train, X_val, y_val
            )

            # Final classification report on val
            y_pred_final = tuned_model.predict(X_val)
            report_str   = classification_report(
                y_val, y_pred_final,
                target_names=[CLASS_NAMES[c] for c in CLASSES],
                zero_division=0
            )
            logging.info(f"\nFull classification report (val):\n{report_str}")

            # Assemble and save JSON report
            report = {
                'best_model':    best_name,
                'phase1_comparison': phase1_results,
                'phase2_tuning':     tuning_result,
                'final_val_f1_weighted': tuning_result.get(
                    'val', {}
                ).get('f1_weighted', phase1_results[best_name]['val']['f1_weighted']),
                'classification_report': report_str,
            }
            os.makedirs(os.path.dirname(self.config.report_file_path), exist_ok=True)
            with open(self.config.report_file_path, 'w') as f:
                json.dump(report, f, indent=2)
            logging.info(f"Model report saved → {self.config.report_file_path}")

            # Save best model
            save_object(
                file_path=self.config.model_file_path,
                obj=tuned_model,
            )
            logging.info(f"Model saved → {self.config.model_file_path}")

            final_f1 = report['final_val_f1_weighted']
            logging.info(f"\n✅ ModelTrainer complete. Best val f1_weighted: {final_f1:.4f}")
            return final_f1, self.config.model_file_path

        except Exception as e:
            raise CustomException(e, sys)






# STANDALONE RUN

if __name__ == "__main__":
    from src.components.data_ingestion     import DataIngestion
    from src.components.data_transformation import DataTransformation

    # Step 1 — Ingest
    ingestion = DataIngestion()
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()

    # Step 2 — Transform
    transformation = DataTransformation()
    train_arr, val_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_path, val_path, test_path
    )

    # Step 3 — Train
    trainer = ModelTrainer()
    best_f1, model_path = trainer.initiate_model_trainer(train_arr, val_arr)

    print(f"\n✅ Training complete")
    print(f"   Best val f1_weighted : {best_f1:.4f}")
    print(f"   Model saved at       : {model_path}")
    print(f"\nNext step: evaluate on TEST set (only touch it once, at the very end):")
    print(f"   from src.utils import load_object")
    print(f"   model = load_object('{model_path}')")
    print(f"   X_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)")
    print(f"   print(model.score(X_test, y_test))")