import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin



#-----------------------------------------------------------------------------------------------------
#                           Fonction pour l'imputation
#-----------------------------------------------------------------------------------------------------


class WeldImputer(BaseEstimator, TransformerMixin):
    """Imputation métier : toutes les statistiques sont apprises sur le train (fit)."""
    
    zero_cols = ["Nickel concentration", "Chromium concentration", "Molybdenum concentration",
                 "Copper concentration", "Vanadium concentration", "Niobium concentration",
                 "Titanium concentration", "Aluminium concentration", "Boron concentration"]
    median_cols = ["Sulphur concentration", "Phosphorus concentration"]
    by_process_cols = ["Oxygen concentration", "Nitrogen concentration"]

    def fit(self, X, y=None):
        X = self._prepare(X)
        self.medians_ = X[self.median_cols].median()
        self.global_proc_ = X[self.by_process_cols].median()
        self.proc_med_ = X.groupby("Type of weld")[self.by_process_cols].median()
        # courant : rapport I / énergie de soudage, par procédé
        self.I_ = X.groupby("Type of weld")["Current"].median()
        self.I_glob_ = X["Current"].median()
        self.U_ = X.groupby("Type of weld")["Voltage"].median()
        self.U_glob_ = X["Voltage"].median()
        self.mode_courant_ = X["courant"].mode()[0]
        return self

    def _prepare(self, X):
        X = X.copy()
        X["Type of weld"] = X["Type of weld"].replace({"ShMA": "MMA"})
        # une polarité + ou - implique du continu
        dc = X["AC or DC"].isna() & X["Electrode positive or negative"].isin(["+", "-"])
        X.loc[dc, "AC or DC"] = "DC"
        X["courant"] = np.select(
            [X["AC or DC"].eq("AC"),
             X["AC or DC"].eq("DC") & X["Electrode positive or negative"].eq("-"),
             X["AC or DC"].eq("DC")],
            ["AC", "DC-", "DC+"], default=None)
        return X.drop(columns=["AC or DC", "Electrode positive or negative"])

    def transform(self, X):
        X = self._prepare(X)
        X[self.zero_cols] = X[self.zero_cols].fillna(0)
        X[self.median_cols] = X[self.median_cols].fillna(self.medians_)
        for c in self.by_process_cols:
            fill = X["Type of weld"].map(self.proc_med_[c]).fillna(self.global_proc_[c])
            X[c] = X[c].fillna(fill)
        X["Current"] = X["Current"].fillna(X["Type of weld"].map(self.I_).fillna(self.I_glob_))
        X["Voltage"] = X["Voltage"].fillna(X["Type of weld"].map(self.U_).fillna(self.U_glob_))
        X["courant"] = X["courant"].fillna(self.mode_courant_)
        return X
    
# ==================================================================
# 2) Fonction d'évaluation unique
# ==================================================================

def metriques(y_true, y_pred):
    return {"RMSE": root_mean_squared_error(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


def evaluer_cv(modele, X, y, folds):
    scores = []
    for idx_tr, idx_va in folds:
        m = clone(modele)                                   # modèle neuf à chaque fold
        m.fit(X.iloc[idx_tr], y.iloc[idx_tr])
        scores.append(metriques(y.iloc[idx_va], m.predict(X.iloc[idx_va])))
    return pd.DataFrame(scores)                             # une ligne par fold


def resumer(scores, metrique="RMSE"):
    s = scores[metrique]
    return {"moyenne": s.mean(),
            "ecart_type": s.std(),
            "SE": s.std() / np.sqrt(len(s)),
            "pire_fold": s.max()}                           # pour le RMSE, le pire = le plus grand


# ==================================================================
# 3) Les modèles et leurs hyperparamètres à régler
#    (chaque fonction reçoit un trial Optuna et renvoie un modèle)
# ==================================================================
RANDOM_STATE = 42

def ridge(trial):
    return make_pipeline(StandardScaler(),                  # standardisation obligatoire
                         Ridge(alpha=trial.suggest_float("alpha", 1e-3, 1e3, log=True)))


def arbre(trial):
    return DecisionTreeRegressor(
        max_depth=trial.suggest_int("max_depth", 2, 15),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        random_state=RANDOM_STATE)


def random_forest(trial):
    return RandomForestRegressor(
        n_estimators=300,                                   # fixé : pas critique
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_float("max_features", 0.3, 1.0),
        n_jobs=-1, random_state=RANDOM_STATE)


def xgboost(trial):
    return XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        n_jobs=-1, random_state=RANDOM_STATE)


