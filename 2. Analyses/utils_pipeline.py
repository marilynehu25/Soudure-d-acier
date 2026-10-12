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
from xgboost import XGBRegressor , XGBClassifier
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, log_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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


#-----------------------------------------------------------------------------------------------------
#                           Fonction pour le feature engineering (CE_IIW)
#-----------------------------------------------------------------------------------------------------

def feature_engineering(X):
    X = X.copy()
    X["CE_IIW"] = (X["Carbon concentration"]
                   + X["Manganese concentration"] / 6
                   + (X["Chromium concentration"] + X["Molybdenum concentration"]
                      + X["Vanadium concentration"]) / 5
                   + (X["Nickel concentration"] + X["Copper concentration"]) / 15)
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



def metriques_cls(y_true, y_pred, y_proba=None):
    """Métriques d'une prédiction de classification (binaire par défaut).

    y_pred : classes prédites (0/1) — métriques dépendantes du seuil.
    y_proba : probabilités de la classe positive — métriques indépendantes du seuil.
    """
    res = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Accuracy_ponderee": balanced_accuracy_score(y_true, y_pred),
        "F1":        f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:                      # métriques indépendantes du seuil
        res["ROC_AUC"] = roc_auc_score(y_true, y_proba)
        res["LogLoss"] = log_loss(y_true, y_proba)
    return res


def evaluer_cv_cls(modele, X, y, folds):
    """Évalue un pipeline de classification par validation croisée.
    Une ligne par fold dans le DataFrame retourné."""
    scores = []
    for idx_tr, idx_va in folds:
        m = clone(modele)                        # modèle neuf à chaque fold
        m.fit(X.iloc[idx_tr], y.iloc[idx_tr])
        y_pred = m.predict(X.iloc[idx_va])
        # probabilités si le modèle en produit (SVM sans proba -> None)
        if hasattr(m, "predict_proba"):
            y_proba = m.predict_proba(X.iloc[idx_va])[:, 1]
        else:
            y_proba = None
        scores.append(metriques_cls(y.iloc[idx_va], y_pred, y_proba))
    return pd.DataFrame(scores)


def resumer_cls(scores, metrique="ROC_AUC"):
    """Résumé d'une métrique de classification (PLUS GRANDE = meilleure)."""
    s = scores[metrique]
    return {"moyenne": s.mean(),
            "ecart_type": s.std(),
            "SE": s.std() / np.sqrt(len(s)),
            "pire_fold": s.min()}               # ⚠️ inversé : ici le pire = le plus petit



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


def logistique(trial):
    """Équivalent du Ridge pour la classification : régression logistique régularisée."""
    return make_pipeline(StandardScaler(),                  # standardisation obligatoire
                         LogisticRegression(
                             C=trial.suggest_float("C", 1e-3, 1e3, log=True),
                             max_iter=1000))


def arbre_cls(trial):
    return DecisionTreeClassifier(
        max_depth=trial.suggest_int("max_depth", 2, 15),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        class_weight="balanced",                            # si classes déséquilibrées
        random_state=RANDOM_STATE)


def random_forest_cls(trial):
    return RandomForestClassifier(
        n_estimators=300,                                   # fixé : pas critique
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_float("max_features", 0.3, 1.0),
        class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE)


def xgboost_cls(trial):
    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        eval_metric="logloss",                              # silencieux + métrique adaptée
        n_jobs=-1, random_state=RANDOM_STATE)


#-----------------------------------------------------------------------------------------------------
#                           Fonction pour l'imputation
#-----------------------------------------------------------------------------------------------------

QUALITY_VARIABLES = [
    "Yield strength",
    "Ultimate tensile strength",
    "Elongation",
    "Reduction of Area",
]

class QualityLabeler:
    """Construit 0 = résistance modérée, 1 = résistance élevée."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, Y):
        Y = Y[QUALITY_VARIABLES]
        self.scaler_ = StandardScaler().fit(Y)
        Y_scaled = self.scaler_.transform(Y)
        self.pca_ = PCA(n_components=2).fit(Y_scaled)
        scores = self.pca_.transform(Y_scaled)
        self.kmeans_ = KMeans(
            n_clusters=2, n_init=50, random_state=self.random_state
        ).fit(scores)

        raw_labels = self.kmeans_.labels_
        scaled_df = pd.DataFrame(Y_scaled, columns=QUALITY_VARIABLES)
        scaled_df["raw_cluster"] = raw_labels
        resistance = (
            scaled_df.groupby("raw_cluster")
            [["Yield strength", "Ultimate tensile strength"]]
            .mean().mean(axis=1)
        )
        self.high_resistance_cluster_ = int(resistance.idxmax())
        return self

    def predict(self, Y):
        Y_scaled = self.scaler_.transform(Y[QUALITY_VARIABLES])
        scores = self.pca_.transform(Y_scaled)
        raw_labels = self.kmeans_.predict(scores)
        return (raw_labels == self.high_resistance_cluster_).astype(int)

    def profile_centers(self):
        centers_scaled = self.pca_.inverse_transform(self.kmeans_.cluster_centers_)
        centers = self.scaler_.inverse_transform(centers_scaled)
        result = pd.DataFrame(centers, columns=QUALITY_VARIABLES)
        result["classe"] = (
            result.index == self.high_resistance_cluster_
        ).astype(int)
        return result.set_index("classe").sort_index()


SIMPLICITE = {  # critère fixé à l'avance : plus petit = plus simple
    "Ridge":         lambda p: -p["alpha"],
    "Arbre":         lambda p: (p["max_depth"], -p["min_samples_leaf"]),
    "Random Forest": lambda p: (p["max_depth"], -p["min_samples_leaf"]),
    "XGBoost":       lambda p: (p["n_estimators"], p["max_depth"], -p["min_child_weight"]),
}



# Critère de simplicité pour la règle 1-SE : plus petit = plus simple
SIMPLICITE_cls = {
    "Logistique":    lambda p: -p["C"],
    "Arbre":         lambda p: (p["max_depth"], -p["min_samples_leaf"]),
    "Random Forest": lambda p: (p["max_depth"], -p["min_samples_leaf"]),
    "XGBoost":       lambda p: (p["n_estimators"], p["max_depth"], -p["min_child_weight"]),
}