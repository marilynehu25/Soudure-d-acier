"""
Fonctions de l'apprentissage semi-supervisé (notebooks 6 et 7).

Organisation
------------
1. Chargement : reconstruction du split train/test de l'équipe + soudures non étiquetées
2. Prétraitement commun (imputation métier de l'équipe + CE_IIW + standardisation + one-hot)
3. Vérification des hypothèses du semi-supervisé (cours CM8)
4. Modèles semi-supervisés et leurs références supervisées
5. Protocole d'évaluation (validation croisée groupée, courbe d'apprentissage)
6. Métriques regroupées, intervalles de confiance, comparaisons appariées
7. Évaluation finale sur le test et sauvegarde (05_reporting)

Toutes les fonctions sont génériques : elles servent pour `Charpy conforme` (notebook 6)
et pour `Profil de résistance` (notebook 7). Seuls changent le masque des lignes étiquetées,
le fichier de split et le JSON du modèle supervisé de référence.
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, kneighbors_graph
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils_pipeline import WeldImputer, feature_engineering, metriques_cls

RANDOM_STATE = 42
ID_COL = "Weld ID"
COL_CAT = ["Type of weld", "courant"]          # variables catégorielles après WeldImputer
CSV = dict(sep=";", encoding="utf-8-sig")       # même format que les CSV de l'équipe

# LabelSpreading émet un RuntimeWarning pour les soudures sans voisin étiqueté (géré par SafeLabelSpreading)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.semi_supervised")


# =====================================================================================================
# 1) Chargement
# =====================================================================================================
def preparer_jeux(df, masque_etiquete, chemin_split, col_cible, colonnes_a_retirer, id_col=ID_COL):
    """Reconstruit EXACTEMENT le découpage train/test du notebook supervisé, et ajoute les non-étiquetées.

    df                : base complète (welddb_processed_vf.csv)
    masque_etiquete   : booléen, lignes de df qui ont une étiquette (ex. df["Charpy conforme"].notna())
    chemin_split      : split_train_test.csv sauvegardé par le notebook supervisé (05_reporting)
    col_cible         : colonne de ce fichier qui contient l'étiquette ("cible" ou "profil")
    colonnes_a_retirer: cibles et colonnes de résultats d'essai (fuite) à retirer des entrées

    Renvoie un dict : X_lab, y_lab, g_lab, train_idx, test_idx, X_unl, g_unl.
    Les non-étiquetées venant d'une publication du TEST sont retirées : le test n'est jamais touché.
    """
    d = df[masque_etiquete].reset_index(drop=True)
    split = pd.read_csv(chemin_split, **CSV)
    if len(split) != len(d) or not (split[id_col].values == d[id_col].values).all():
        raise ValueError("Le split sauvegardé ne correspond plus aux données : relancer le notebook supervisé.")

    a_retirer = [c for c in colonnes_a_retirer + [id_col] if c in df.columns]
    X_lab = d.drop(columns=a_retirer)
    y_lab = split[col_cible].astype(int)
    g_lab = d[id_col]
    train_idx = np.where(split["jeu"].values == "Train")[0]
    test_idx = np.where(split["jeu"].values == "Test")[0]

    unl = df[~masque_etiquete]
    sources_test = set(g_lab.iloc[test_idx])
    garde = ~unl[id_col].isin(sources_test)
    return {
        "X_lab": X_lab, "y_lab": y_lab, "g_lab": g_lab,
        "train_idx": train_idx, "test_idx": test_idx,
        "X_unl": unl.loc[garde].drop(columns=a_retirer).reset_index(drop=True),
        "g_unl": unl.loc[garde, id_col].reset_index(drop=True),
        "n_unl_exclues_test": int((~garde).sum()),
    }


def charger_reference(chemin_json):
    """Lit le modele_retenu.json du notebook supervisé (nom + hyperparamètres du modèle retenu)."""
    with open(chemin_json, encoding="utf-8") as f:
        return json.load(f)


def modele_depuis_json(nom, params, random_state=RANDOM_STATE):
    """Reconstruit le classifieur retenu par l'équipe, avec les mêmes définitions que utils_pipeline."""
    if nom == "XGBoost (défaut)":
        return XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=random_state)
    if nom == "XGBoost":
        return XGBClassifier(**params, eval_metric="logloss", n_jobs=-1, random_state=random_state)
    if nom == "Random Forest":
        return RandomForestClassifier(n_estimators=300, **params, class_weight="balanced",
                                      n_jobs=-1, random_state=random_state)
    if nom == "Arbre":
        return DecisionTreeClassifier(**params, class_weight="balanced", random_state=random_state)
    if nom == "Logistique":
        return LogisticRegression(**params, max_iter=1000)   # la standardisation est faite dans PretraitementSSL
    raise ValueError(f"Modèle inconnu : {nom}")


# =====================================================================================================
# 2) Prétraitement commun
# =====================================================================================================
class PretraitementSSL(BaseEstimator, TransformerMixin):
    """Imputation métier de l'équipe (WeldImputer) + CE_IIW + standardisation + one-hot.

    - mêmes étapes que les notebooks supervisés, pour que les comparaisons soient justes ;
    - standardisation ajoutée : label spreading et CEM reposent sur des DISTANCES (sans effet sur les arbres) ;
    - fit sur S ∪ U (entrées étiquetées et non étiquetées) : utiliser les entrées non étiquetées pour
      apprendre l'imputation est déjà légitime (elles sont disponibles) et commun à toutes les méthodes,
      ce qui isole l'effet de l'ALGORITHME semi-supervisé.
    """

    def fit(self, X, y=None):
        self.imputer_ = WeldImputer().fit(X)
        Z = feature_engineering(self.imputer_.transform(X))
        self.col_num_ = [c for c in Z.columns if c not in COL_CAT]
        self.col_cat_ = [c for c in Z.columns if c in COL_CAT]
        self.encodeur_ = ColumnTransformer([
            ("num", StandardScaler(), self.col_num_),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.col_cat_),
        ]).fit(Z)
        return self

    def transform(self, X):
        return self.encodeur_.transform(feature_engineering(self.imputer_.transform(X)))

    def noms_variables(self):
        return list(self.encodeur_.get_feature_names_out())


# =====================================================================================================
# 3) Vérification des hypothèses (cours CM8)
# =====================================================================================================
def part_sources_communes(g_lab, g_unl):
    """Part des non-étiquetées issues d'une publication qui contient aussi des étiquetées."""
    return float(g_unl.isin(set(g_lab)).mean())


def figure_superposition_acp(Z_lab, Z_unl, y_lab, titre, noms_classes=("classe 0", "classe 1"), ax=None):
    """Hypothèse de lissité / clusters : les non-étiquetées sont-elles dans la même zone que les étiquetées ?"""
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(np.vstack([Z_lab, Z_unl]))
    P_lab, P_unl = pca.transform(Z_lab), pca.transform(Z_unl)
    ax = ax or plt.gca()
    ax.scatter(P_unl[:, 0], P_unl[:, 1], s=8, c="lightgrey", label=f"non étiquetées ({len(P_unl)})")
    for k, (couleur, nom) in enumerate(zip(("#c0392b", "#2c7fb8"), noms_classes)):
        m = np.asarray(y_lab) == k
        ax.scatter(P_lab[m, 0], P_lab[m, 1], s=12, c=couleur, alpha=.75, label=f"{nom} ({m.sum()})")
    ev = pca.explained_variance_ratio_
    ax.set(xlabel=f"Z1 ({ev[0]:.0%})", ylabel=f"Z2 ({ev[1]:.0%})", title=titre)
    ax.legend(fontsize=8)
    return ax


def test_lissite(Z_lab, y_lab, g_lab, k=5):
    """Hypothèse de lissité : les voisins étiquetés d'une soudure ont-ils la même classe ?

    Compare l'accord observé avec l'accord attendu par hasard (p0² + p1²), et le recalcule
    en n'autorisant que des voisins d'AUTRES publications (cas réel du GroupKFold).
    """
    y = np.asarray(y_lab)
    g = np.asarray(g_lab)
    nn = NearestNeighbors(n_neighbors=min(len(y) - 1, 60)).fit(Z_lab)
    _, idx = nn.kneighbors(Z_lab)
    idx = idx[:, 1:]                                              # on enlève le point lui-même
    accord_tous = (y[idx[:, :k]] == y[:, None]).mean()
    accords_autres = []
    for i in range(len(y)):
        autres = idx[i][g[idx[i]] != g[i]][:k]
        if len(autres):
            accords_autres.append((y[autres] == y[i]).mean())
    p = np.bincount(y) / len(y)
    return pd.Series({
        "accord attendu par hasard": float((p ** 2).sum()),
        f"accord avec les {k} plus proches voisins": float(accord_tous),
        f"accord avec les {k} plus proches voisins d'AUTRES publications": float(np.mean(accords_autres)),
    })


def test_ponts(Z_train_lab, Z_unl, Z_test):
    """Les non-étiquetées « comblent-elles les trous » entre le train et le test ?
    Distance de chaque soudure de test à la plus proche étiquetée, puis à la plus proche (étiquetée ou non)."""
    d_lab = NearestNeighbors(n_neighbors=1).fit(Z_train_lab).kneighbors(Z_test)[0].ravel()
    d_tout = NearestNeighbors(n_neighbors=1).fit(np.vstack([Z_train_lab, Z_unl])).kneighbors(Z_test)[0].ravel()
    return pd.Series({
        "distance médiane test -> étiquetée la plus proche": float(np.median(d_lab)),
        "distance médiane test -> soudure la plus proche (étiquetée ou non)": float(np.median(d_tout)),
        "part du test dont la plus proche voisine est NON étiquetée": float((d_tout < d_lab - 1e-12).mean()),
    })


# =====================================================================================================
# 4) Modèles
# =====================================================================================================
class SafeLabelSpreading(LabelSpreading):
    """LabelSpreading dont les probabilités indéfinies (soudure sans aucun voisin informé : 0/0)
    sont remplacées par la proportion des classes, au lieu de prédire arbitrairement la classe 0."""

    def fit(self, X, y):
        super().fit(X, y)
        y = np.asarray(y)
        self.prior_ = np.array([(y[y != -1] == c).mean() for c in self.classes_])
        return self

    def predict_proba(self, X):
        p = super().predict_proba(X)
        mauvais = ~np.isfinite(p).all(axis=1) | (p.sum(axis=1) == 0)
        p[mauvais] = self.prior_
        self.n_isoles_ = int(mauvais.sum())
        return p

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class SelfTrainingEquilibre(BaseEstimator, ClassifierMixin):
    """Self-training avec pseudo-étiquetage ÉQUILIBRÉ PAR CLASSE.

    Défaut du self-training classique (constaté sur Charpy avec l'ancienne définition) : les prédictions
    les plus sûres sont presque toutes de la classe majoritaire, donc seules des pseudo-étiquettes
    majoritaires sont ajoutées et le modèle renforce son propre biais (biais de confirmation).
    Ici, à chaque itération, on ajoute pour CHAQUE classe les soudures les plus sûrement prédites dans
    cette classe, en respectant la proportion des classes des vraies étiquettes, et seulement si la
    probabilité dépasse `seuil`.
    """

    def __init__(self, estimator, part_par_iter=0.1, seuil=0.7, max_iter=10):
        self.estimator = estimator
        self.part_par_iter = part_par_iter
        self.seuil = seuil
        self.max_iter = max_iter

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        lab = y != -1
        X_l, y_l, X_u = X[lab], y[lab], X[~lab]
        self.classes_ = np.unique(y_l)
        prior = np.array([(y_l == c).mean() for c in self.classes_])
        n_iter_total = max(1, int(self.part_par_iter * len(X_u)))
        self.n_pseudo_par_classe_ = {int(c): 0 for c in self.classes_}
        for _ in range(self.max_iter):
            if len(X_u) == 0:
                break
            m = clone(self.estimator).fit(X_l, y_l)
            proba = m.predict_proba(X_u)
            choisis = []
            for j, c in enumerate(self.classes_):
                cand = np.where((proba.argmax(axis=1) == j) & (proba[:, j] >= self.seuil))[0]
                n_c = max(1, int(round(n_iter_total * prior[j])))
                choisis.append(cand[np.argsort(-proba[cand, j])][:n_c])
                self.n_pseudo_par_classe_[int(c)] += len(choisis[-1])
            choisis = np.concatenate(choisis).astype(int)
            if len(choisis) == 0:
                break
            X_l = np.vstack([X_l, X_u[choisis]])
            y_l = np.concatenate([y_l, self.classes_[proba[choisis].argmax(axis=1)]])
            X_u = np.delete(X_u, choisis, axis=0)
        self.estimator_ = clone(self.estimator).fit(X_l, y_l)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X))


class SemiSupervisedCEM(BaseEstimator, ClassifierMixin):
    """CEM semi-supervisé (Celeux & Govaert ; cours CM8) : une gaussienne par classe.
    E : probabilités a posteriori des non-étiquetées ; C : affectation à la classe la plus probable ;
    M : ré-estimation des proportions, moyennes et covariances sur S ∪ U. y = -1 pour les non-étiquetées.
    max_iter = 0  ->  même modèle appris sur les étiquetées seules (référence supervisée)."""

    def __init__(self, max_iter=50, covariance_type="diag", reg=1e-3):
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.reg = reg

    def _m_step(self, X, y):
        self.priors_, self.means_, self.covs_ = [], [], []
        for k in self.classes_:
            Xk = X[y == k]
            self.priors_.append(len(Xk) / len(X))
            self.means_.append(Xk.mean(axis=0))
            var = Xk.var(axis=0) + self.reg if len(Xk) > 1 else np.ones(X.shape[1])
            cov = np.diag(var) if self.covariance_type == "diag" else np.cov(Xk.T) + self.reg * np.eye(X.shape[1])
            self.covs_.append(cov)

    def _log_joint(self, X):
        out = []
        for p, m, c in zip(self.priors_, self.means_, self.covs_):
            diff = X - m
            _, logdet = np.linalg.slogdet(c)
            maha = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(c), diff)
            out.append(np.log(p) - 0.5 * (logdet + maha + X.shape[1] * np.log(2 * np.pi)))
        return np.column_stack(out)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        lab = y != -1
        self.classes_ = np.unique(y[lab])
        self._m_step(X[lab], y[lab])                               # initialisation sur S
        y_u, self.n_iter_ = None, 0
        for it in range(self.max_iter):
            if lab.all():
                break
            nouv = self.classes_[self._log_joint(X[~lab]).argmax(axis=1)]   # étapes E et C
            if y_u is not None and np.array_equal(nouv, y_u):
                break
            y_u = nouv
            y_tout = y.copy()
            y_tout[~lab] = y_u
            self._m_step(X, y_tout)                                # étape M sur S ∪ U
            self.n_iter_ = it + 1
        return self

    def predict_proba(self, X):
        lj = self._log_joint(np.asarray(X, dtype=float))
        lj -= lj.max(axis=1, keepdims=True)
        p = np.exp(lj)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def methodes_classification(modele_reference, nom_reference, n_voisins=10, n_composantes=10):
    """Les méthodes comparées, par paires (supervisé de référence -> semi-supervisé de la même famille).

    Renvoie (methodes, paires) :
      methodes = {nom: (estimateur, utilise_les_non_etiquetees)}
      paires   = [(famille, nom_supervise, nom_semi_supervise)]
    """
    ref = f"{nom_reference} (supervisé, retenu par l'équipe)"
    methodes = {
        ref:                              (clone(modele_reference), False),
        "Self-training classique":        (SelfTrainingClassifier(clone(modele_reference), threshold=0.9, max_iter=10), True),
        "Self-training équilibré":        (SelfTrainingEquilibre(clone(modele_reference)), True),
        "kNN (supervisé)":                (KNeighborsClassifier(n_neighbors=n_voisins, weights="distance"), False),
        "Label spreading":                (SafeLabelSpreading(kernel="knn", n_neighbors=n_voisins, alpha=0.2, max_iter=200), True),
        "Gaussiennes (supervisé)":        (make_pipeline(PCA(n_components=n_composantes, random_state=RANDOM_STATE),
                                                         SemiSupervisedCEM(max_iter=0)), False),
        "CEM semi-supervisé":             (make_pipeline(PCA(n_components=n_composantes, random_state=RANDOM_STATE),
                                                         SemiSupervisedCEM(max_iter=50)), True),
    }
    paires = [
        ("Discriminante (self-training)", ref, "Self-training classique"),
        ("Discriminante (self-training)", ref, "Self-training équilibré"),
        ("Graphe", "kNN (supervisé)", "Label spreading"),
        ("Générative", "Gaussiennes (supervisé)", "CEM semi-supervisé"),
    ]
    return methodes, paires


# =====================================================================================================
# 5) Protocole d'évaluation
# =====================================================================================================
def sous_echantillonner(y, frac, rng, minimum=3):
    """Garde une fraction des étiquettes, stratifiée par classe (au moins `minimum` par classe)."""
    idx = np.arange(len(y))
    if frac >= 1:
        return idx
    garde = [rng.choice(idx[y == c], size=min((y == c).sum(), max(minimum, int(round(frac * (y == c).sum())))),
                        replace=False) for c in np.unique(y)]
    return np.sort(np.concatenate(garde))


def evaluer_ssl_cv(jeux, methodes, fractions=(0.1, 0.25, 0.5, 1.0), n_tirages=3, n_folds=5,
                   random_state=RANDOM_STATE, verbose=True):
    """Validation croisée groupée sur le TRAIN de l'équipe (le test n'est jamais utilisé).

    Pour chaque pli, chaque fraction d'étiquettes et chaque tirage :
      - S = étiquetées des plis d'entraînement (sous-échantillonnées selon la fraction) ;
      - U = non-étiquetées hors publications du pli de validation + étiquettes CACHÉES de S ;
      - prétraitement appris sur S ∪ U ; chaque méthode est entraînée puis évaluée sur le pli de validation.
    Renvoie toutes les prédictions hors-pli (une ligne par soudure de validation, méthode, fraction, tirage).
    """
    X_tr = jeux["X_lab"].iloc[jeux["train_idx"]].reset_index(drop=True)
    y_tr = jeux["y_lab"].iloc[jeux["train_idx"]].to_numpy()
    g_tr = jeux["g_lab"].iloc[jeux["train_idx"]].reset_index(drop=True)
    X_unl, g_unl = jeux["X_unl"], jeux["g_unl"]

    folds = list(StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                 .split(X_tr, y_tr, g_tr))
    lignes = []
    for f, (i_fit, i_val) in enumerate(folds):
        U_pli = X_unl[~g_unl.isin(set(g_tr.iloc[i_val])).values]
        for frac in fractions:
            for t in range(1 if frac >= 1 else n_tirages):
                rng = np.random.default_rng(random_state + 100 * f + t)
                garde = sous_echantillonner(y_tr[i_fit], frac, rng)
                cache = np.setdiff1d(np.arange(len(i_fit)), garde)
                X_S, y_S = X_tr.iloc[i_fit[garde]], y_tr[i_fit[garde]]
                X_U = pd.concat([U_pli, X_tr.iloc[i_fit[cache]]], ignore_index=True)

                prep = PretraitementSSL().fit(pd.concat([X_S, X_U], ignore_index=True))
                Z_S, Z_U, Z_val = prep.transform(X_S), prep.transform(X_U), prep.transform(X_tr.iloc[i_val])
                for nom, (est, semi) in methodes.items():
                    m = clone(est)
                    if semi:
                        m.fit(np.vstack([Z_S, Z_U]), np.concatenate([y_S, -np.ones(len(Z_U), dtype=int)]))
                    else:
                        m.fit(Z_S, y_S)
                    proba = m.predict_proba(Z_val)[:, 1]
                    lignes.append(pd.DataFrame({
                        "fold": f, "fraction": frac, "tirage": t, "methode": nom, "semi": semi,
                        "groupe": g_tr.iloc[i_val].values, "y": y_tr[i_val],
                        "y_pred": m.predict(Z_val), "proba": proba,
                        "n_S": len(y_S), "n_U": len(Z_U)}))
        if verbose:
            print(f"pli {f + 1}/{n_folds} terminé")
    return pd.concat(lignes, ignore_index=True)


# =====================================================================================================
# 6) Métriques
# =====================================================================================================
def scores_cls(y, y_pred, proba=None):
    """Métriques de l'équipe (metriques_cls) + F1 de la classe 0 et F1 macro.
    Avec des classes déséquilibrées, le F1 « par défaut » ne regarde que la classe 1 :
    on ajoute le F1 de la classe 0 et le F1 macro (les deux classes à égalité)."""
    res = metriques_cls(y, y_pred, proba if proba is not None and len(np.unique(y)) > 1 else None)
    res["F1_classe_0"] = f1_score(y, y_pred, pos_label=0, zero_division=0)
    res["F1_macro"] = f1_score(y, y_pred, average="macro", zero_division=0)
    return res


def scores_regroupes(preds):
    """Métriques sur les prédictions hors-pli REGROUPÉES (pooled), par méthode, fraction et tirage, puis
    moyennées sur les tirages. On regroupe car un pli de validation peut contenir très peu de soudures
    de la classe minoritaire (Forman & Scholz, 2010)."""
    lignes = [{"methode": m, "fraction": f, "tirage": t, **scores_cls(p["y"], p["y_pred"], p["proba"])}
              for (m, f, t), p in preds.groupby(["methode", "fraction", "tirage"], sort=False)]
    return pd.DataFrame(lignes).groupby(["methode", "fraction"], sort=False).mean(numeric_only=True).drop(columns="tirage")


def _metrique(y, y_pred, proba, metrique):
    """Calcule UNE métrique (rapide, pour le bootstrap)."""
    if metrique in ("Accuracy_ponderee", "F1_classe_0", "F1_macro", "F1", "Recall", "Precision", "Accuracy"):
        return scores_cls(y, y_pred)[metrique]
    return scores_cls(y, y_pred, proba)[metrique]


def _indices_par_groupe(preds):
    return {g: np.asarray(idx) for g, idx in preds.groupby("groupe").indices.items()}


def ic_bootstrap(preds, metrique="Accuracy_ponderee", n_boot=500, random_state=RANDOM_STATE):
    """IC à 95 % par bootstrap sur les PUBLICATIONS (tirage de publications entières avec remise)."""
    preds = preds.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    idx_g = _indices_par_groupe(preds)
    noms = list(idx_g)
    y, yp, pr = preds["y"].to_numpy(), preds["y_pred"].to_numpy(), preds["proba"].to_numpy()
    vals = []
    for _ in range(n_boot):
        ii = np.concatenate([idx_g[g] for g in rng.choice(noms, size=len(noms), replace=True)])
        vals.append(_metrique(y[ii], yp[ii], pr[ii], metrique))
    return np.percentile(vals, [2.5, 97.5])


def gain_apparie(preds, nom_sup, nom_semi, metrique="Accuracy_ponderee", fraction=1.0,
                 n_boot=1000, random_state=RANDOM_STATE):
    """Gain semi-supervisé − supervisé, avec IC à 95 % par bootstrap APPARIÉ sur les publications :
    les deux méthodes sont évaluées sur les MÊMES publications tirées, donc le bruit commun s'annule
    (bien plus puissant que comparer deux intervalles séparés)."""
    p = preds[preds["fraction"] == fraction]
    cle = ["tirage", "fold", "groupe"]
    a = p[p["methode"] == nom_sup].sort_values(cle, kind="stable").reset_index(drop=True)
    b = p[p["methode"] == nom_semi].sort_values(cle, kind="stable").reset_index(drop=True)
    ya, pa, qa = a["y"].to_numpy(), a["y_pred"].to_numpy(), a["proba"].to_numpy()
    yb, pb, qb = b["y"].to_numpy(), b["y_pred"].to_numpy(), b["proba"].to_numpy()
    gain = _metrique(yb, pb, qb, metrique) - _metrique(ya, pa, qa, metrique)
    rng = np.random.default_rng(random_state)
    idx_g = _indices_par_groupe(a)                 # mêmes lignes dans a et b (même ordre)
    noms = list(idx_g)
    vals = []
    for _ in range(n_boot):
        ii = np.concatenate([idx_g[g] for g in rng.choice(noms, size=len(noms), replace=True)])
        vals.append(_metrique(yb[ii], pb[ii], qb[ii], metrique) - _metrique(ya[ii], pa[ii], qa[ii], metrique))
    vals = np.array(vals)
    bas, haut = np.percentile(vals, [2.5, 97.5])
    return {"gain": gain, "IC95_bas": bas, "IC95_haut": haut, "P(gain > 0)": float((vals > 0).mean())}


def tableau_gains(preds, paires, metrique="Accuracy_ponderee", fractions=None, n_boot=1000):
    """Gains appariés de chaque paire (famille), pour chaque fraction d'étiquettes."""
    fractions = fractions or sorted(preds["fraction"].unique())
    lignes = []
    for famille, sup, semi in paires:
        for f in fractions:
            lignes.append({"famille": famille, "supervisé": sup, "semi-supervisé": semi, "fraction": f,
                           **gain_apparie(preds, sup, semi, metrique, f, n_boot)})
    return pd.DataFrame(lignes)


# =====================================================================================================
# Graphiques
# =====================================================================================================
COULEURS_FAMILLES = {"Discriminante (self-training)": "C0", "Graphe": "C2", "Générative": "C3"}


def courbe_apprentissage(scores, paires, metrique="Accuracy_ponderee", titre="", ax=None):
    """Score selon la part d'étiquettes : pointillés = supervisé, trait plein = semi-supervisé (même couleur)."""
    ax = ax or plt.gca()
    deja = set()
    styles = ["-", "-."]
    for i, (famille, sup, semi) in enumerate(paires):
        c = COULEURS_FAMILLES.get(famille, f"C{i}")
        for nom, ls, mk in ((sup, "--", "o"), (semi, styles[sum(1 for p in paires[:i] if p[0] == famille) % 2], "s")):
            if nom in deja:
                continue
            deja.add(nom)
            s = scores.xs(nom, level="methode")[metrique]
            ax.plot(s.index * 100, s.values, ls=ls, marker=mk, color=c, label=nom)
    ax.set_xscale("log")
    fr = sorted(scores.index.get_level_values("fraction").unique())
    ax.set_xticks([f * 100 for f in fr]); ax.set_xticklabels([f"{int(f * 100)} %" for f in fr])
    ax.set(xlabel="Part des étiquettes d'entraînement conservées", ylabel=metrique, title=titre)
    ax.legend(fontsize=7, ncol=2)
    return ax


def matrices_confusion(preds, methodes, noms_classes=("classe 0", "classe 1"), fraction=1.0):
    """Matrices de confusion des prédictions hors-pli regroupées."""
    n = len(methodes)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.2 * ((n + 1) // 2), 7.5))
    for ax, nom in zip(axes.ravel(), methodes):
        p = preds[(preds["methode"] == nom) & (preds["fraction"] == fraction)]
        cm = confusion_matrix(p["y"], p["y_pred"], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=noms_classes, yticklabels=noms_classes)
        ax.set_title(nom, fontsize=9); ax.set_xlabel("prédit"); ax.set_ylabel("réel")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    plt.tight_layout()
    return fig


# =====================================================================================================
# 7) Évaluation finale sur le test et sauvegarde
# =====================================================================================================
def evaluation_test(jeux, methodes, noms, n_boot=1000, random_state=RANDOM_STATE):
    """Entraîne les méthodes `noms` sur TOUT le train étiqueté (+ toutes les non-étiquetées hors test),
    puis prédit le test UNE SEULE FOIS. Renvoie (récapitulatif, prédictions, modèles entraînés, prétraitement)."""
    X_S = jeux["X_lab"].iloc[jeux["train_idx"]]
    y_S = jeux["y_lab"].iloc[jeux["train_idx"]].to_numpy()
    X_T = jeux["X_lab"].iloc[jeux["test_idx"]]
    y_T = jeux["y_lab"].iloc[jeux["test_idx"]].to_numpy()
    g_T = jeux["g_lab"].iloc[jeux["test_idx"]].to_numpy()
    prep = PretraitementSSL().fit(pd.concat([X_S, jeux["X_unl"]], ignore_index=True))
    Z_S, Z_U, Z_T = prep.transform(X_S), prep.transform(jeux["X_unl"]), prep.transform(X_T)

    recap, preds, modeles = {}, [], {}
    for nom in noms:
        est, semi = methodes[nom]
        m = clone(est)
        if semi:
            m.fit(np.vstack([Z_S, Z_U]), np.concatenate([y_S, -np.ones(len(Z_U), dtype=int)]))
        else:
            m.fit(Z_S, y_S)
        p = pd.DataFrame({"methode": nom, "position_dans_d": jeux["test_idx"], "groupe": g_T,
                          "y": y_T, "y_pred": m.predict(Z_T), "proba": m.predict_proba(Z_T)[:, 1]})
        sc = scores_cls(p["y"], p["y_pred"], p["proba"])
        bas, haut = ic_bootstrap(p.assign(fraction=1.0), "Accuracy_ponderee", n_boot, random_state)
        recap[nom] = {**sc, "Accuracy_ponderee IC95 bas": bas, "Accuracy_ponderee IC95 haut": haut}
        preds.append(p)
        modeles[nom] = m
    return pd.DataFrame(recap).T, pd.concat(preds, ignore_index=True), modeles, prep


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    return o


def enregistrer_ssl(dossier, infos, preds_cv, scores, gains, recap_test, preds_test, modele_final, prep, hypotheses):
    """Sauvegarde dans 05_reporting/semi_supervise/<cible>/ (même logique que les notebooks supervisés) :
    - modele_retenu_ssl.json : méthode retenue, hyperparamètres, référence supervisée, scores CV et test ;
    - CSV : prédictions hors-pli, scores par méthode et fraction, gains appariés, test, hypothèses ;
    - modele_final_ssl.joblib : prétraitement + modèle retenu, réutilisables sans réentraîner."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    preds_cv.to_csv(dossier / "cv_predictions.csv", index=False, **CSV)
    scores.reset_index().to_csv(dossier / "cv_scores_par_methode_et_fraction.csv", index=False, **CSV)
    gains.to_csv(dossier / "cv_gains_apparies.csv", index=False, **CSV)
    recap_test.to_csv(dossier / "test_recap_metriques.csv", **CSV)
    preds_test.to_csv(dossier / "test_predictions.csv", index=False, **CSV)
    hypotheses.to_csv(dossier / "verification_hypotheses.csv", header=["valeur"], **CSV)
    joblib.dump({"pretraitement": prep, "modele": modele_final}, dossier / "modele_final_ssl.joblib")
    with open(dossier / "modele_retenu_ssl.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(infos), f, indent=2, ensure_ascii=False)
    return dossier
