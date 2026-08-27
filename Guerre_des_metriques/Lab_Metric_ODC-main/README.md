# Detection de fraude bancaire

TP Masterclass Classification : analyse des metriques d'un modele de detection
de fraude bancaire (matrice de confusion, precision/rappel, ROC, PR, cout
metier, equite) sur 200 transactions.

## Prerequis

- Python 3.11 ou plus recent (Windows, macOS, Linux)

## Installation

Avec [uv](https://docs.astral.sh/uv/) :

```bash
uv sync
```

Ou avec pip :

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Utilisation

```bash
uv run python main.py
# ou, si l'environnement est deja active :
python main.py
```

Le script lit `data/input/dataset_fraude.csv` et genere un tableau de bord
interactif dans `data/output/dashboard.html`, ouvert automatiquement dans le
navigateur par defaut.

## Structure

```
data/
  input/    dataset source (verse dans le depot)
  output/   dashboard genere (ignore par git)
main.py     script d'analyse et de generation du tableau de bord
```
