import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
# charge des données (réduites depuis listings.csv)
df = pd.read_csv("data/listings.csv.gz", compression="gzip")
# préparation du sous-ensemble de colonnes utiles
features = [
    "price", "neighbourhood_cleansed", "room_type", "latitude", "longitude",
    "accommodates", "bedrooms", "bathrooms_text", "number_of_reviews"
]
df = df[features].copy()
# nettoyage du prix
df = df[df["price"].notna()]
df["price"] = df["price"].replace("[\$,]", "", regex=True).astype(float)
df = df[df["price"] > 0]
# suppression lignes trop vides
df = df.dropna(thresh=6)
# définition X et y
X = df.drop(columns="price")
y = df["price"]
# colonnes numériques / catégorielles
num_cols = ["accommodates", "latitude", "longitude", "number_of_reviews"]
cat_cols = ["neighbourhood_cleansed", "room_type"]
# pipeline numérique avec discrétisation des coordonnées
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
# pipeline catégoriel
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
# préprocessing global
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])
# pipeline complet avec PCA optionnel
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=10))
])
# transformation des données
X_processed = full_pipeline.fit_transform(X)
# Séparation entrainement / test
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
# Définition des modèles
models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "DecisionTree": DecisionTreeRegressor(max_depth=5)
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append((name, mse, r2))
results_df = pd.DataFrame(results, columns=["Model", "MSE", "R2"])
print("\n\n=== Résultats des modèles ===")
print(results_df.sort_values("R2", ascending=False))
