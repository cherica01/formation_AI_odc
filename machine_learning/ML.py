import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Chargement des données
df_train = pd.read_csv("train.csv")
df_val = pd.read_csv("valid.csv")

# 2. Nettoyage du Train Set
df_train_clean = df_train.dropna(subset=["surface_m2"]).copy()

mediane_nb_pieces = df_train_clean["nb_pieces"].median()
df_train_clean["nb_pieces"] = df_train_clean["nb_pieces"].fillna(mediane_nb_pieces)

colonnes_a_imputer = ["age_annees", "distance_centre_km", "price"]
for col in colonnes_a_imputer:
    if df_train_clean[col].isnull().sum() > 0:
        mediane_val = df_train_clean[col].median()
        df_train_clean[col] = df_train_clean[col].fillna(mediane_val)

colonnes = ["surface_m2", "nb_pieces", "age_annees", "distance_centre_km", "price"]
for col in colonnes:
    Q1 = df_train_clean[col].quantile(0.25)
    Q3 = df_train_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR
    df_train_clean = df_train_clean[(df_train_clean[col] >= borne_inf) & (df_train_clean[col] <= borne_sup)]

# 3. Nettoyage du Validation Set
df_val_clean = df_val.dropna(subset=["surface_m2"]).copy()
df_val_clean["nb_pieces"] = df_val_clean["nb_pieces"].fillna(mediane_nb_pieces)

for col in colonnes_a_imputer:
    if df_val_clean[col].isnull().sum() > 0:
        mediane_val = df_train_clean[col].median()
        df_val_clean[col] = df_val_clean[col].fillna(mediane_val)

# 4. Séparation X et y
X_train = df_train_clean.drop(columns=["price"])
y_train = df_train_clean["price"]

X_val = df_val_clean.drop(columns=["price"])
y_val = df_val_clean["price"]

# 5. Standardisation (Création de X_train_scaled)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 6. Entraînement de la Régression Linéaire
model = LinearRegression()
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

# 8. Graphiques de diagnostic des résidus
residuals_train = y_train - y_pred_train

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_pred_train, residuals_train, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Valeurs prédites (y_pred)")
plt.ylabel("Résidus (y_train - y_pred)")
plt.title("Homoscédasticité des résidus")

plt.subplot(1, 2, 2)
stats.probplot(residuals_train, dist="norm", plot=plt)
plt.title("QQ-Plot des résidus")

plt.tight_layout()
plt.show()
