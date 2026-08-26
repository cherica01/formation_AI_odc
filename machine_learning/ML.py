import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

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

    df_train_clean = df_train_clean[(df_train_clean[col] >= borne_inf) & (df_train_clean[col] <= borne_sup)]
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


# ============================================================
# 8. DIAGNOSTIC DE GAUSS-MARKOV
# On refait un modèle OLS "de référence" (Ridge n'est pas concerné
# par Gauss-Markov car il est volontairement biaisé)
# ============================================================

# statsmodels a besoin qu'on ajoute nous-même une colonne de "1"
# pour représenter la constante (l'ordonnée à l'origine)
X_train_sm = sm.add_constant(X_train_scaled)
ols_model = sm.OLS(y_train, X_train_sm).fit()

# Résumé statistique complet
print("\n--- Résumé du modèle OLS ---")
print(ols_model.summary())

residus = ols_model.resid
valeurs_predites = ols_model.fittedvalues

# ------------------------------------------------------------
# 8.1 Linéarité + Exogénéité : graphique résidus vs prédictions
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(valeurs_predites, residus, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Valeurs prédites")
plt.ylabel("Résidus")
plt.title("Résidus vs Valeurs prédites")
plt.show()

# ------------------------------------------------------------
# 8.2 Homoscédasticité : test de Breusch-Pagan
# ------------------------------------------------------------
bp_test = het_breuschpagan(residus, X_train_sm)
labels = ["LM stat", "LM p-value", "F stat", "F p-value"]
print("\n--- Test de Breusch-Pagan (homoscédasticité) ---")
print(dict(zip(labels, bp_test)))

# ------------------------------------------------------------
# 8.3 Normalité des résidus : QQ-plot + test de Shapiro-Wilk
# ------------------------------------------------------------
sm.qqplot(residus, line='45', fit=True)
plt.title("QQ-plot des résidus")
plt.show()

stat, p_value = stats.shapiro(residus)
print(f"\n--- Test de Shapiro-Wilk (normalité) ---")
print(f"stat={stat:.4f}, p-value={p_value:.4f}")

# ------------------------------------------------------------
# 8.4 Indépendance des erreurs : Durbin-Watson
# ------------------------------------------------------------
dw_stat = durbin_watson(residus)
print(f"\n--- Durbin-Watson (autocorrélation) ---")
print(f"Statistique : {dw_stat:.3f}")

# ------------------------------------------------------------
# 8.5 Multicolinéarité : VIF
# ------------------------------------------------------------
vif_data = pd.DataFrame()
vif_data["variable"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]
print("\n--- VIF (multicolinéarité) ---")
print(vif_data)