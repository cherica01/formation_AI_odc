import pandas as pd

df = pd.read_csv("train.csv")
print(f"--- Chargement terminé : {len(df)} lignes chargées ---")

missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
resultat = pd.DataFrame({
    "Valeurs_manquantes": missing,
    "Pourcentage": missing_percent
})
print("\n--- Diagnostic initial des valeurs manquantes ---")
print(resultat)

colonnes = [
    "surface_m2",
    "nb_pieces",
    "age_annees",
    "distance_centre_km",
    "price"
]

print("\n--- Détection des outliers (sur données brutes) ---")
for col in colonnes:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR

    outliers = df[(df[col] < borne_inf) | (df[col] > borne_sup)]
    print(f"{col} : {len(outliers)} outliers")

df_clean = df.dropna(subset=["surface_m2"]).copy()
print(f"\n--- Suppression des nuls dans surface_m2 ---")
print(f"Nombre de lignes restantes : {len(df_clean)}")

mediane_nb_pieces = df_clean["nb_pieces"].median()
df_clean["nb_pieces"] = df_clean["nb_pieces"].fillna(mediane_nb_pieces)
print(f"\n--- Imputation des nuls dans nb_pieces ---")
print(f"Médiane utilisée : {mediane_nb_pieces}")
print(f"Valeurs manquantes restantes dans nb_pieces : {df_clean['nb_pieces'].isnull().sum()}")

colonnes_a_imputer = ["age_annees", "distance_centre_km", "price"]

for col in colonnes_a_imputer:
    if df_clean[col].isnull().sum() > 0:
        mediane_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(mediane_val)
        print(f"\n--- Imputation des nuls dans {col} ---")
        print(f"Médiane utilisée pour {col} : {mediane_val}")
        print(f"Valeurs manquantes restantes dans {col} : {df_clean[col].isnull().sum()}")
    else:
        print(f"\n--- Aucune valeur manquante détectée dans {col} ---")

print("\n--- Nettoyage des outliers sur l'ensemble des colonnes ---")
for col in colonnes:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1

    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR

    lignes_avant = len(df_clean)
    df_clean = df_clean[(df_clean[col] >= borne_inf) & (df_clean[col] <= borne_sup)]
    lignes_supprimees = lignes_avant - len(df_clean)
    
    print(f"Suppression des outliers pour {col} : {lignes_supprimees} lignes retirées")

missing_final = df_clean.isnull().sum()
resultat_final = pd.DataFrame({
    "Valeurs_manquantes": missing_final,
    "Pourcentage": (missing_final / len(df_clean)) * 100
})

print("\n--- Diagnostic FINAL du jeu de données nettoyé ---")
print(resultat_final)
print(f"\nTaille finale du DataFrame : {len(df_clean)} lignes (initialement {len(df)})")