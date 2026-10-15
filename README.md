# Prédiction de la qualité de soudures d'acier

Projet d'apprentissage automatique, CentraleSupélec (octobre 2026).
**Équipe :** Marilyne Hu, Martin Potier, Sandrine Daniel.

La qualité d'une soudure dépend de la composition du métal, du procédé et du traitement thermique, un savoir encore surtout détenu par des experts. Ce projet cherche à **prédire la qualité d'une soudure à partir des paramètres connus avant soudage**, puis à en tirer des **recommandations**.

**Données :** [Weld Metal Database](https://www.phasetrans.msm.cam.ac.uk/map/data/materials/welddb-b.html), University of Cambridge. Elle contient 1 651 soudures et 44 variables (composition, procédé, traitement thermique, propriétés mécaniques, microstructure).

---

## 🎯 Ce que nous prédisons

| Cible | Définition | Étiquetées | Modèle retenu | Accuracy pondérée (test) |
|---|---|---|---|---|
| **Charpy conforme** (ténacité) | Au moins 47 J à −40 °C (EN ISO 2560-A), avec interpolation de T(47 J) entre T(28 J) et T(100 J) | 569 (82 % conformes) | XGBoost, toutes les variables | **0,82** |
| **Profil de résistance** | 2 profils (modéré / élevé) obtenus par ACP + K-means sur les 4 propriétés de traction | 650 (35 % élevés) | Arbre de décision de profondeur 2 | **0,85** |

**Le semi-supervisé** (self-training, label spreading, CEM) **n'améliore pas** ces modèles quand toutes les étiquettes sont utilisées. Le self-training apporte un **gain significatif quand les étiquettes sont rares** (+0,02 à +0,03 avec 25 % des étiquettes, pour le profil de résistance).

**Recommandation principale :** garder un carbone équivalent `CE_IIW` ≤ 0,40 pour la ténacité. La zone `CE_IIW` ≈ 0,37-0,40 offre le meilleur compromis entre résistance et ténacité. Les autres recommandations (manganèse, azote, micro-alliages, oxygène, titane) sont dans le notebook 8 et le rapport.

---

## 📓 Les notebooks (`2. Analyses/`), à lancer dans l'ordre

| # | Notebook | Contenu |
|---|---|---|
| 1 | `1. Traitement de données.ipynb` | Lecture des fichiers bruts, conversion des valeurs mal formatées (`<L` → L/2, intervalles → milieu), passage des ppm en % massique |
| 2 | `2. Pré-processing des données.ipynb` | Valeurs manquantes, valeurs aberrantes, création de `Charpy conforme` et de `CE_IIW`, corrélations, ACP |
| 3 | `3_Prediction_par_target_5_cibles.ipynb` | Prédiction supervisée de `Charpy conforme` : logistique, arbre, forêt aléatoire, XGBoost réglés par Optuna, avec (B) et sans (A) sélection de variables |
| 4 | `4. Construction d'une variable de qualité par clustering.ipynb` | Construction du profil de résistance (ACP + K-means) |
| 5 | `5. Prédiction de profil de resistance par classification.ipynb` | Prédiction supervisée du profil de résistance |
| 6 | `6. Apprentissage semi-supervisé charpy.ipynb` | Semi-supervisé sur `Charpy conforme` : vérification des hypothèses, validation croisée imbriquée, comparaison au modèle de l'équipe |
| 7 | `7. Apprentissage semi-supervisé profil resistance.ipynb` | Semi-supervisé sur le profil de résistance, même protocole |
| 8 | `8. Comparatif final et recommandations.ipynb` | Toutes les méthodes sur le même test, choix de l'approche, recommandations (importance des variables, dépendance partielle, taux observés) |

**Modules :**
- `utils_pipeline.py` : imputation (`WeldImputer`), carbone équivalent, espaces de recherche Optuna, construction du profil (`QualityLabeler`) ;
- `utils_semi_supervise.py` : méthodes semi-supervisées, tests d'hypothèses, protocole d'évaluation, sauvegarde.

---

## 🔬 Choix méthodologiques clés

- **Découpage par publication.** Le `Weld ID` désigne une série publiée. Les soudures d'une même série se ressemblent, donc train/test et validation croisée sont **groupés par publication** (`StratifiedGroupKFold`). Une même publication n'est jamais à la fois dans le train et dans le test.
- **Pas de fuite d'information.** L'imputation, le profil de résistance et le réglage des hyperparamètres sont appris sur le train uniquement. Le test n'est utilisé qu'une fois.
- **Métrique principale : l'accuracy pondérée.** Les cibles sont déséquilibrées : un modèle qui prédit toujours « conforme » obtient déjà un F1 de 0,91, mais une accuracy pondérée de 0,5 seulement.
- **Intervalles de confiance** obtenus par bootstrap des publications. Les gains du semi-supervisé sont **appariés** (mêmes plis que le modèle de l'équipe) et ne sont retenus que si leur IC95 exclut 0.

---

## 📁 Organisation du dépôt

```
.
├── _old_/                    # Corbeille : notebooks et scripts abandonnés
├── 0. Annexes/
│   ├── Sujet-soudures.pdf    # Sujet du projet
│   ├── welddb.pdf            # Documentation de la base
│   ├── Référence documentaire/
│   └── Rapport/              # Rapport final (rapport.tex, rapport.pdf, figures/)
├── 1. Données/
│   ├── 01_raw/               # Données brutes (welddb.data, welddb.info)
│   ├── 02_intermediate/      # Données traitées (welddb_processed_vf.csv) et dictionnaire des variables
│   └── 05_reporting/         # Résultats enregistrés
│       ├── prediction_Charpy conforme/   # Supervisé Charpy (méthodes A et B)
│       ├── profil_resistance/            # Supervisé profil (méthodes A et B)
│       ├── semi_supervise/
│       │   ├── charpy/                   # Semi-supervisé Charpy
│       │   └── profil_resistance/        # Semi-supervisé profil
│       └── comparatif_final/             # Comparatif et recommandations
├── 2. Analyses/              # Notebooks 1 à 8 et modules utils_*.py
└── 3. Résultats/figures/     # Figures (ssl_*, final_*)
```

Chaque dossier de résultats contient un **JSON du modèle retenu** (hyperparamètres, scores, recommandation), les **prédictions** (validation croisée et test) et le **modèle entraîné** (`.joblib`). Les dossiers semi-supervisés contiennent aussi les prédictions sur toutes les soudures **non étiquetées** (`predictions_non_etiquetees.csv`).

---

## 🛠️ Installation de l'environnement

Depuis la racine du projet, créer un environnement virtuel :

```bash
python3 -m venv .venv
```

L'activer sous macOS ou Linux :

```bash
source .venv/bin/activate
```

Sous Windows (PowerShell) :

```powershell
.venv\Scripts\Activate.ps1
```

Installer ensuite les packages nécessaires :

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Les notebooks 3 et 5 utilisent aussi **Optuna** (`pip install optuna`), qui n'est pas encore dans `requirements.txt`.

Pour quitter l'environnement virtuel :

```bash
deactivate
```

---

## 🤝 Travail collaboratif

### Règles

- **Une branche par tâche** (ex. `feature/nettoyage-donnees`, `fix/erreur-split`).
- **Une branche par personne** pour les tests personnels (ex. `test/prenom`).
- **On ne travaille jamais directement sur `main`** : tout passe par une *Pull Request* sur GitHub, relue par un autre membre avant la fusion.

### Commandes à retenir

```bash
# 1. Se mettre à jour
git switch main
git pull

# 2. Créer sa branche
git switch -c feature/ma-fonctionnalite

# 3. Travailler, puis enregistrer
git status                 # voir ce qui a changé
git add fichier.py         # ou « git add . » pour tout ajouter
git commit -m "feat: ajoute ma fonctionnalité"

# 4. Envoyer sa branche sur GitHub
git push -u origin feature/ma-fonctionnalite
# → puis ouvrir une Pull Request sur GitHub

# 5. Après la fusion de la PR, faire le ménage
git switch main
git pull
git branch -d feature/ma-fonctionnalite
```

### Récupérer les mises à jour de `main` dans sa branche

```bash
git switch main
git pull
git switch feature/ma-fonctionnalite
git merge main
```

---

## 👥 Répartition du travail

| Membre | Contributions principales |
|---|---|
| **Marilyne Hu** | Mise en place du dépôt, nettoyage et imputation, fusion des notebooks, pipeline générique, prédiction de `Charpy conforme` et du profil de résistance |
| **Martin Potier** | Environnement, premier passage sur les données, modèles et pipelines, sélection de variables, construction du profil de résistance (ACP + K-means) |
| **Sandrine Daniel** | Analyse descriptive et ACP, apprentissage semi-supervisé, comparatif final et recommandations |
| **Tous** | Préprocessing (mis en commun en réunion), choix du feature engineering, rapport |
