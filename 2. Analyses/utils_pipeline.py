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
