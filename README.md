# Projet Apprentissage Automatique : Soudure d'acier

## 📁 Organisation du dépôt

```
.
├── _old_/                  # Corbeille : données, scripts ou notebooks abandonnés
├── 0. Annexes/             # Sujets, documentation et rapport
├── 1. Données/
│   ├── 00_models/          # Paramètres des modèles
│   ├── 01_raw/             # Données brutes
│   ├── 02_intermediate/    # Données traitées
│   ├── 03_model_input/     # Données d'entrée des modèles (découpées en train / test / validation)
│   ├── 04_model_output/    # Données de sortie des modèles
│   └── 05_reporting/       # Données finales
├── 2. Analyses/            # Notebooks
└── 3. Résultats/           # Sauvegarde de tous les résultats
```

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

Pour quitter l'environnement virtuel :

```bash
deactivate
```

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
