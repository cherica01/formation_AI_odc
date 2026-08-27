import pandas as pd

df = pd.read_csv('dataset_fraude.csv')
df['y_pred'] = (df['y_proba'] >= 0.5 ).astype(int)
