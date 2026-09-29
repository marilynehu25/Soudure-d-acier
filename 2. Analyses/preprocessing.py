"""
Préprocessing commun du projet « Soudure d'acier ».

Toutes les décisions (et leur justification) sont dans le notebook
`2. Préprocessing et analyse descriptive.ipynb`. Ce module ne fait que les appliquer.

Utilisation type (dans un notebook de `2. Analyses`) :

    from preprocessing import load_xy, build_preprocessor
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor

    X, y, groups = load_xy("Yield strength")
    pipe = Pipeline([("prep", build_preprocessor()), ("model", RandomForestRegressor())])
    pipe.fit(X_train, y_train)      # fit + transform sur le train
    pipe.predict(X_test)            # transform seulement sur le test (paramètres appris sur le train)

Règle : ce qui APPREND des données (seuil de NaN, imputation, standardisation, encodage)
est dans le pipeline et n'est donc calculé que sur le train. Les règles fixes (nettoyage texte)
y sont aussi, pour qu'une nouvelle soudure brute soit traitée automatiquement.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------------------
# Chemins et rôle des colonnes
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PATH_PROCESSED = ROOT / "1. Données" / "02_intermediate" / "welddb_processed.csv"

TARGETS = [
    "Yield strength",
    "Ultimate tensile strength",
    "Elongation",
    "Reduction of Area",
    "Charpy impact toughness",
]
TEST_CONDITION = ["Charpy temperature"]  # entrée uniquement pour la cible Charpy

CHEMISTRY = [
    "Carbon concentration", "Silicon concentration", "Manganese concentration",
    "Sulphur concentration", "Phosphorus concentration", "Nickel concentration",
    "Chromium concentration", "Molybdenum concentration", "Vanadium concentration",
    "Copper concentration", "Cobalt concentration", "Tungsten concentration",
    "Oxygen concentration", "Titanium concentration", "Nitrogen concentration",
    "Aluminium concentration", "Boron concentration", "Niobium concentration",
    "Tin concentration", "Arsenic concentration", "Antimony concentration",
]
PROCESS_NUM = [
    "Current", "Voltage", "Heat input", "Interpass temperature",
    "Post weld heat treatment temperature", "Post weld heat treatment time",
]
PROCESS_CAT_RAW = ["AC or DC", "Electrode positive or negative", "Type of weld"]

# Résultats du soudage ou colonnes quasi vides : jamais utilisées comme entrées
UNUSABLE = [
    "Hardness", "50 % FATT",
    "Primary ferrite in microstructure", "Ferrite with second phase",
    "Acicular ferrite", "Martensite", "Ferrite with carbide aggreagate",
]
ID_COL = "Weld ID"

FEATURES_RAW = CHEMISTRY + PROCESS_NUM + PROCESS_CAT_RAW

# Colonnes après nettoyage
NUMERIC = CHEMISTRY + PROCESS_NUM + TEST_CONDITION
BINARY = ["is_DC", "electrode_positive"]
CATEGORICAL = ["Type of weld"]
# Éléments avec beaucoup de valeurs « <L » (> 30 lignes) : on garde un indicateur 0/1
LOD_FLAG_ELEMENTS = ["Vanadium", "Titanium", "Aluminium", "Boron", "Niobium"]

# Incohérences d'unités (voir notebook, section 2.a)
UNIT_FIXES = {
    "Vanadium concentration": {"<5": "<0.0005"},   # ppm saisis dans une colonne en %
    "Titanium concentration": {"<0.01": "<100"},   # % saisis dans une colonne en ppm
    "Aluminium concentration": {"<0.01": "<100"},
}


# --------------------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------------------
def source_from_id(weld_id):
    """Publication d'origine : début du Weld ID jusqu'à l'année ('Evans-AlTi-1994-450/1ch1' -> 'Evans-AlTi-1994')."""
    m = re.match(r"^(.*?\d{4})", str(weld_id))
    return m.group(1) if m else re.split(r"[-/ ]", str(weld_id))[0]


def load_raw(path=PATH_PROCESSED):
    """Base complète, lue en texte (les valeurs '<0.01', '150-200'… sont gardées telles quelles)."""
    return pd.read_csv(path, sep=";", dtype=object)


def load_xy(target, path=PATH_PROCESSED):
    """
    Renvoie (X brut, y, groupes) pour UNE cible, en gardant seulement les lignes où la cible est connue.
    C'est la seule étape faite hors pipeline : un pipeline sklearn ne peut pas supprimer de lignes.
    - X : entrées brutes (texte), à passer au pipeline
    - y : cible numérique
    - groups : source bibliographique, pour GroupKFold / GroupShuffleSplit
    """
    if target not in TARGETS:
        raise ValueError(f"cible inconnue : {target}. Choix : {TARGETS}")
    raw = load_raw(path)
    y = pd.to_numeric(raw[target], errors="coerce")
    keep = y.notna()
    features = FEATURES_RAW + (TEST_CONDITION if target == "Charpy impact toughness" else [])
    X = raw.loc[keep, features].reset_index(drop=True)
    groups = raw.loc[keep, ID_COL].map(source_from_id).reset_index(drop=True)
    return X, y[keep].reset_index(drop=True), groups


def load_unlabeled(target, path=PATH_PROCESSED):
    """Lignes SANS la cible (pour le semi-supervisé) : (X brut, groupes)."""
    raw = load_raw(path)
    miss = pd.to_numeric(raw[target], errors="coerce").isna()
    features = FEATURES_RAW + (TEST_CONDITION if target == "Charpy impact toughness" else [])
    return (raw.loc[miss, features].reset_index(drop=True),
            raw.loc[miss, ID_COL].map(source_from_id).reset_index(drop=True))


# --------------------------------------------------------------------------------------
# Transformers
# --------------------------------------------------------------------------------------
class WeldCleaner(BaseEstimator, TransformerMixin):
    """
    Règles FIXES (rien n'est appris) : texte -> nombres.
    - '<L' -> L * lod_factor (0.5 = substitution L/2, Hornung & Reed 1990) + indicateur '<élément>_below_LOD'
    - corrections d'unités (UNIT_FIXES)
    - azote '67tot33res' -> 67 (teneur totale)
    - intervalle '150-200' -> 175 (milieu)
    - 'ShMA' -> 'MMA' (même procédé d'après la doc)
    - 'AC or DC' -> is_DC (0/1), 'Electrode positive or negative' -> electrode_positive (0/1)
    """

    def __init__(self, lod_factor=0.5, lod_flags=True):
        self.lod_factor = lod_factor
        self.lod_flags = lod_flags

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy().astype(object)

        for col, mapping in UNIT_FIXES.items():
            if col in X:
                X[col] = X[col].replace(mapping)

        for col in [c for c in CHEMISTRY if c in X]:
            s = X[col]
            below = s.astype(str).str.startswith("<")
            if below.any():
                X.loc[below, col] = s[below].astype(str).str[1:].astype(float) * self.lod_factor
            element = col.replace(" concentration", "")
            if self.lod_flags and element in LOD_FLAG_ELEMENTS:
                X[f"{element}_below_LOD"] = below.astype(int)

        if "Nitrogen concentration" in X:
            X["Nitrogen concentration"] = X["Nitrogen concentration"].astype(str).str.extract(
                r"^([\d.]+)", expand=False)

        if "Interpass temperature" in X:
            s = X["Interpass temperature"].astype(str)
            m = s.str.fullmatch(r"([\d.]+)-([\d.]+)")
            if m.any():
                X.loc[m, "Interpass temperature"] = s[m].str.split("-", expand=True).astype(float).mean(axis=1)

        if "Type of weld" in X:
            X["Type of weld"] = X["Type of weld"].replace({"ShMA": "MMA"})
        if "AC or DC" in X:
            X["is_DC"] = X.pop("AC or DC").map({"DC": 1, "AC": 0})
        if "Electrode positive or negative" in X:
            X["electrode_positive"] = X.pop("Electrode positive or negative").map({"+": 1, "-": 0})

        num = [c for c in X.columns if c not in CATEGORICAL]
        X[num] = X[num].apply(pd.to_numeric, errors="coerce")
        return X

    def get_feature_names_out(self, input_features=None):
        return None


class DropSparseColumns(BaseEstimator, TransformerMixin):
    """
    Supprime les colonnes numériques dont le taux de NaN (calculé sur le TRAIN) dépasse `threshold`.
    C'est une étape qui apprend des données -> elle doit être dans le pipeline.
    """

    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        candidates = [c for c in X.columns if c in NUMERIC]
        rate = X[candidates].isna().mean()
        self.nan_rate_ = rate
        self.columns_dropped_ = rate[rate > self.threshold].index.tolist()
        return self

    def transform(self, X):
        return pd.DataFrame(X).drop(columns=self.columns_dropped_, errors="ignore")


# Sélecteurs de colonnes : appelés au moment du fit, donc s'adaptent aux colonnes supprimées
def select_numeric(X):
    return [c for c in X.columns if c in NUMERIC]


def select_binary(X):
    return [c for c in X.columns if c in BINARY or c.endswith("_below_LOD")]


def select_categorical(X):
    return [c for c in X.columns if c in CATEGORICAL]


def build_preprocessor(nan_threshold=0.8, imputer="median", scale=True, lod_factor=0.5):
    """
    Pipeline de préprocessing complet, à mettre avant le modèle.
    - imputer="median"    : SimpleImputer univarié (baseline robuste)
    - imputer="iterative" : IterativeImputer(sample_posterior=True) = imputation par régression stochastique (cours CM0)
    Dans les deux cas add_indicator=True : une colonne 0/1 par variable ayant des manquants (hypothèse MAR/MNAR).
    """
    if imputer == "median":
        num_imputer = SimpleImputer(strategy="median", add_indicator=True)
    elif imputer == "iterative":
        num_imputer = IterativeImputer(sample_posterior=True, max_iter=20, random_state=0, add_indicator=True)
    else:
        raise ValueError("imputer doit valoir 'median' ou 'iterative'")

    num_steps = [("impute", num_imputer)] + ([("scale", StandardScaler())] if scale else [])
    encode = ColumnTransformer([
        ("num", Pipeline(num_steps), select_numeric),
        ("bin", SimpleImputer(strategy="most_frequent"), select_binary),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), select_categorical),
    ])
    return Pipeline([
        ("clean", WeldCleaner(lod_factor=lod_factor)),
        ("drop_sparse", DropSparseColumns(threshold=nan_threshold)),
        ("encode", encode),
    ])
