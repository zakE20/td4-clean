# TD4 - Machine Learning : Prédiction du prix des logements Airbnb à Paris

## Objectif
Utiliser Scikit-learn pour construire un modèle de machine learning permettant de prédire le **prix d’un logement Airbnb** à Paris, à partir de données du fichier `listings.csv`.

---

## Étapes réalisées

### Pré-traitement :
- Nettoyage de la colonne `price` (conversion de chaîne → float)
- Suppression des lignes avec trop de valeurs manquantes
- Imputation des données manquantes restantes
- Standardisation des colonnes numériques
- Encodage One-Hot des colonnes catégorielles :
  - `neighbourhood_cleansed`
  - `room_type`
- Réduction de dimensionnalité avec PCA (10 composantes)

### Données utilisées :
- `accommodates`, `latitude`, `longitude`, `number_of_reviews`
- `neighbourhood_cleansed`, `room_type`

### Modèles testés :
- Régression Linéaire
- Lasso
- ElasticNet
- K plus proches voisins (KNN)
- Arbre de Décision

---

## Résultats

| Modèle             | MSE (erreur quadratique moyenne) | R² Score |
|--------------------|-------------------------------|----------|
| ElasticNet         | 376574.86                     | 0.0456   |
| Lasso              | 376646.17                     | 0.0455   |
| Linear Regression  | 376656.05                     | 0.0454   |
| Decision Tree      | 392168.05                     | 0.0061   |
| KNN                | 410518.99                     | -0.0404  |

---

## Conclusion

Le modèle **ElasticNet** a donné les meilleurs résultats avec un score **R² ≈ 0.046**, ce qui reste assez faible. Cela s'explique probablement par :
- Peu de variables explicatives utilisées
- Données bruitées ou non linéaires
- Importance potentielle de variables non incluses (surface, équipements, etc.)

Des pistes d'amélioration :
- Enrichir le jeu de données avec de nouvelles variables
- Réaliser un tuning plus fin des hyperparamètres
- Tester d'autres modèles non-linéaires (Random Forest, Gradient Boosting)

---

## Fichiers

- `td4.py` : code du pipeline complet
- `README.md` : résumé du TD

---

## 👨 Auteur

Zakaria LAACHIRI
L3 MIAGE – UCA DS4H 2025
