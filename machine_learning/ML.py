import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 1. Chargement des données
df_train = pd.read_csv("train.csv")
df_val = pd.read_csv("valid.csv")

# 2. Nettoyage initial et Imputation
df_train_clean = df_train.dropna(subset=["surface_m2"]).copy()
df_val_clean = df_val.dropna(subset=["surface_m2"]).copy()

mediane_nb_pieces = df_train_clean["nb_pieces"].median()
df_train_clean["nb_pieces"] = df_train_clean["nb_pieces"].fillna(mediane_nb_pieces)
df_val_clean["nb_pieces"] = df_val_clean["nb_pieces"].fillna(mediane_nb_pieces)

colonnes_a_imputer = ["age_annees", "distance_centre_km", "price"]
for col in colonnes_a_imputer:
    mediane_val = df_train_clean[col].median()
    df_train_clean[col] = df_train_clean[col].fillna(mediane_val)
    df_val_clean[col] = df_val_clean[col].fillna(mediane_val)

# 3. Filtrage des Outliers (Sur Train ET Validation avec les bornes du Train)
colonnes = ["surface_m2", "nb_pieces", "age_annees", "distance_centre_km", "price"]
for col in colonnes:
    Q1 = df_train_clean[col].quantile(0.25)
    Q3 = df_train_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR
    
    # Filtrage du train
    df_train_clean = df_train_clean[(df_train_clean[col] >= borne_inf) & (df_train_clean[col] <= borne_sup)]
    # Filtrage de la validation pour évaluation équitable
    df_val_clean = df_val_clean[(df_val_clean[col] >= borne_inf) & (df_val_clean[col] <= borne_sup)]

# 4. Séparation X et y
X_train = df_train_clean.drop(columns=["price"])
y_train = df_train_clean["price"]

X_val = df_val_clean.drop(columns=["price"])
y_val = df_val_clean["price"]

# 5. Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 6. Entraînement Ridge
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# 7. Prédictions et Évaluation
y_pred_train = model.predict(X_train_scaled)
y_pred_val = model.predict(X_val_scaled)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

r2_train = r2_score(y_train, y_pred_train)
r2_val = r2_score(y_val, y_pred_val)

print("--- Évaluation du modèle ---")
print(f"Train RMSE : {rmse_train:.2f} | Train R² : {r2_train:.4f}")
print(f"Val RMSE   : {rmse_val:.2f} | Val R²   : {r2_val:.4f}")