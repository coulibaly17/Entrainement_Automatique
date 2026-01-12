# Application Machine Learning - Régression & Classification

Application Streamlit permettant d'effectuer des analyses de régression et de classification sur des données personnalisées.

## 🚀 Fonctionnalités

- **📁 Téléchargement de données** : Importez vos propres fichiers CSV
- **📉 Régression** : Prédisez des valeurs continues avec différents algorithmes
  - Régression Linéaire
  - Arbre de Décision
  - Forêt Aléatoire
  - Gradient Boosting
- **📊 Classification** : Prédisez des catégories avec les mêmes algorithmes

## 📋 Prérequis

- Python 3.8+
- Bibliothèques listées dans `requirements.txt`

## 🛠 Installation

1. Clonez le dépôt
2. Créez un environnement virtuel :
   ```
   python -m venv env
   ```
3. Activez l'environnement :
   - Windows : `env\Scripts\activate`
   - Mac/Linux : `source env/bin/activate`
4. Installez les dépendances :
   ```
   pip install -r requirements.txt
   ```

## 🚀 Lancement

```
streamlit run app.py
```

## 📊 Jeux de données inclus

- `heart.csv` : Données sur les maladies cardiaques (classification)
- `house_prices_datasetss_2000.csv` : Données immobilières (régression)

## 📝 Utilisation

1. Téléversez votre fichier CSV dans l'onglet "📁 Téléchargement"
2. Choisissez entre régression ou classification
3. Sélectionnez votre variable cible
4. L'application entraîne automatiquement les modèles et affiche les résultats
5. Utilisez l'interface pour faire des prédictions personnalisées

## 📊 Métriques d'évaluation

### Régression
- MAE (Erreur Absolue Moyenne)
- MSE (Erreur Quadratique Moyenne)
- R² (Coefficient de Détermination)

### Classification
- Précision
- Rappel
- F1-Score
- Exactitude

## 📝 Notes

- L'application gère automatiquement les variables catégorielles
- Les modèles sont entraînés avec une répartition 80/20 (train/test)
- La graine aléatoire est fixée pour la reproductibilité
