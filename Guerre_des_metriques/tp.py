import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
df = pd.read_csv('dataset_fraude.csv')
df['y_pred'] = (df['y_proba'] >= 0.5 ).astype(int)
print(df['y_pred'])

tn,fp,fn,tp = confusion_matrix(df['y_true'],df['y_pred']).ravel()
print(f"VP={tp},VN={tn},FP={fp},FN={fn}")

acc = accuracy_score(df['y_true'], df['y_pred'])
prec = precision_score(df['y_true'], df['y_pred'])
rec = recall_score(df['y_true'], df['y_pred'])
f1 = f1_score(df['y_true'], df['y_pred'])

print(f"Accuracy={acc:.3f}, Précision={prec:.3f}, Rappel={rec:.3f}, F1={f1:.3f}")

naive_accuracy = (df['y_true'] == 0).mean()
print(f"Accuracy du modèle naïf = {naive_accuracy:.3f}")

for seuil in [0.3, 0.5, 0.7]:
    y_pred = (df['y_proba'] >= seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(df['y_true'], y_pred).ravel()
    prec = precision_score(df['y_true'], y_pred, zero_division=0)
    rec = recall_score(df['y_true'], y_pred, zero_division=0)
    print(f"seuil={seuil}: VP={tp} FP={fp} FN={fn} | précision={prec:.3f} rappel={rec:.3f}")
    from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(df['y_true'], df['y_proba'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="steelblue", label=f"Courbe ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Modèle aléatoire")
plt.xlabel("Taux de Faux Positifs (FP / (FP+VN))")
plt.ylabel("Taux de Vrais Positifs = Rappel")
plt.title("Courbe ROC")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)   # <- sauvegarde le fichier
plt.show()                               # <- AFFICHE la fenêtre avec le graphique
plt.close()