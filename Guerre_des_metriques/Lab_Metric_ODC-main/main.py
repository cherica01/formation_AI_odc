from pathlib import Path
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as pyo
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    matthews_corrcoef,
)

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "data" / "input" / "dataset_fraude.csv"
OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_HTML = OUTPUT_DIR / "dashboard.html"

if not INPUT_CSV.exists():
    raise SystemExit(f"Fichier introuvable : {INPUT_CSV}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# palette (voir data-viz skill : categorical slots 1/2/4/7 + status)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
YELLOW = "#eda100"
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
PAGE = "#f2f1ee"

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

df = pd.read_csv(INPUT_CSV)

SEUIL = 0.5
df["y_pred"] = (df["y_proba"] >= SEUIL).astype(int)

# Q2 - matrice de confusion
tn, fp, fn, tp = confusion_matrix(df.y_true, df.y_pred).ravel()

# Q3 - metriques au seuil par defaut
accuracy = accuracy_score(df.y_true, df.y_pred)
precision = precision_score(df.y_true, df.y_pred)
recall = recall_score(df.y_true, df.y_pred)
f1 = f1_score(df.y_true, df.y_pred)

# Q4 - modele naif (toujours "non-fraude")
y_naif = np.zeros(len(df), dtype=int)
accuracy_naif = accuracy_score(df.y_true, y_naif)
recall_naif = recall_score(df.y_true, y_naif, zero_division=0)

# Q5 - effet du seuil
seuils_test = [0.3, 0.5, 0.7]
precisions_seuils, recalls_seuils = [], []
for s in seuils_test:
    yp = (df.y_proba >= s).astype(int)
    precisions_seuils.append(precision_score(df.y_true, yp, zero_division=0))
    recalls_seuils.append(recall_score(df.y_true, yp, zero_division=0))

# Q6 - ROC
fpr, tpr, _ = roc_curve(df.y_true, df.y_proba)
auc_roc = roc_auc_score(df.y_true, df.y_proba)

# Q7 - precision/rappel
prc_precisions, prc_recalls, _ = precision_recall_curve(df.y_true, df.y_proba)
auc_pr = average_precision_score(df.y_true, df.y_proba)

# Q8 - MCC
mcc = matthews_corrcoef(df.y_true, df.y_pred)

# Q9 - cout metier et seuil optimal
COUT_FP = 15
COUT_FN = 500


def cout_total(y_true, y_pred):
    tn_, fp_, fn_, tp_ = confusion_matrix(y_true, y_pred).ravel()
    return fp_ * COUT_FP + fn_ * COUT_FN


cout_defaut = cout_total(df.y_true, df.y_pred)
seuils_range = np.arange(0.05, 0.96, 0.01)
couts = np.array([cout_total(df.y_true, (df.y_proba >= s).astype(int)) for s in seuils_range])
seuil_opt = seuils_range[couts.argmin()]
cout_opt = couts.min()

# Q10 - equite entre regions
recalls_region = {}
for region in sorted(df.region.unique()):
    sub = df[df.region == region]
    yp = (sub.y_proba >= SEUIL).astype(int)
    recalls_region[region] = recall_score(sub.y_true, yp, zero_division=0)


def base_layout(height=340):
    return dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_SECONDARY, size=12),
        margin=dict(l=48, r=24, t=16, b=44),
        height=height,
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK_MUTED)),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK_MUTED)),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11, color=INK_SECONDARY)),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(family=FONT, color=INK)),
    )


def to_div(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "responsive": True})


# --- Q2 : matrice de confusion ---
cm_values = [[tn, fp], [fn, tp]]
cm_labels = [["VN", "FP"], ["FN", "VP"]]
fig_cm = go.Figure(go.Heatmap(
    z=cm_values,
    x=["Predit 0", "Predit 1"],
    y=["Reel 0", "Reel 1"],
    colorscale=[[0, "#cde2fb"], [0.35, "#5598e7"], [0.65, "#256abf"], [1, "#0d366b"]],
    showscale=False,
    hovertemplate="%{y} / %{x} : %{z}<extra></extra>",
))
for i in range(2):
    for j in range(2):
        value = cm_values[i][j]
        light_text = value > tn / 2
        fig_cm.add_annotation(
            x=j, y=i, text=f"{cm_labels[i][j]}<br><b>{value}</b>",
            showarrow=False, font=dict(color="white" if light_text else INK, size=14),
        )
fig_cm.update_layout(**base_layout(300))
fig_cm.update_yaxes(autorange="reversed")

# --- Q3 : metriques au seuil 0.5 ---
metrics_names = ["Accuracy", "Precision", "Rappel", "F1-score"]
metrics_values = [accuracy, precision, recall, f1]
metrics_colors = [BLUE, VIOLET, ORANGE, YELLOW]
fig_metrics = go.Figure(go.Bar(
    x=metrics_names, y=metrics_values, marker_color=metrics_colors,
    text=[f"{v:.2f}" for v in metrics_values], textposition="outside",
    textfont=dict(color=INK_SECONDARY), width=0.5,
    hovertemplate="%{x} : %{y:.3f}<extra></extra>",
))
fig_metrics.update_layout(**base_layout(300))
fig_metrics.update_yaxes(range=[0, 1.15])

# --- Q4 : piege de l'accuracy ---
fig_trap = go.Figure()
fig_trap.add_bar(name="Accuracy", x=["Modele naif", "Modele testé"], y=[accuracy_naif, accuracy],
                  marker_color=BLUE, width=0.32, hovertemplate="Accuracy (%{x}) : %{y:.3f}<extra></extra>")
fig_trap.add_bar(name="Rappel", x=["Modele naif", "Modele testé"], y=[recall_naif, recall],
                  marker_color=ORANGE, width=0.32, hovertemplate="Rappel (%{x}) : %{y:.3f}<extra></extra>")
fig_trap.update_layout(barmode="group", bargap=0.35, bargroupgap=0.1, **base_layout(300))
fig_trap.update_yaxes(range=[0, 1.05])

# --- Q5 : precision / rappel selon le seuil ---
fig_seuil = go.Figure()
fig_seuil.add_bar(name="Precision", x=[str(s) for s in seuils_test], y=precisions_seuils,
                   marker_color=VIOLET, width=0.32, hovertemplate="Precision (seuil %{x}) : %{y:.3f}<extra></extra>")
fig_seuil.add_bar(name="Rappel", x=[str(s) for s in seuils_test], y=recalls_seuils,
                   marker_color=ORANGE, width=0.32, hovertemplate="Rappel (seuil %{x}) : %{y:.3f}<extra></extra>")
fig_seuil.update_layout(barmode="group", bargap=0.35, bargroupgap=0.1, **base_layout(300))
fig_seuil.update_xaxes(title=dict(text="Seuil de decision", font=dict(color=INK_MUTED)))
fig_seuil.update_yaxes(range=[0, 1.05])

# --- Q6 : courbe ROC ---
fig_roc = go.Figure()
fig_roc.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=BLUE, width=2.5),
                     fill="tozeroy", fillcolor="rgba(42,120,214,0.08)",
                     hovertemplate="FPR %{x:.2f} - TPR %{y:.2f}<extra></extra>")
fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=INK_MUTED, width=1.5), hoverinfo="skip")
fig_roc.add_annotation(x=0.62, y=0.22, text=f"AUC-ROC = <b>{auc_roc:.3f}</b>", showarrow=False,
                        font=dict(color=INK_SECONDARY, size=12))
fig_roc.update_layout(showlegend=False, **base_layout(300))
fig_roc.update_xaxes(title=dict(text="Taux de faux positifs", font=dict(color=INK_MUTED)), range=[0, 1])
fig_roc.update_yaxes(title=dict(text="Taux de vrais positifs", font=dict(color=INK_MUTED)), range=[0, 1.02])

# --- Q7 : courbe precision-rappel ---
fig_pr = go.Figure()
fig_pr.add_scatter(x=prc_recalls, y=prc_precisions, mode="lines", line=dict(color=BLUE, width=2.5),
                    fill="tozeroy", fillcolor="rgba(42,120,214,0.08)",
                    hovertemplate="Rappel %{x:.2f} - Precision %{y:.2f}<extra></extra>")
fig_pr.add_annotation(x=0.3, y=0.15, text=f"PR-AUC = <b>{auc_pr:.3f}</b>", showarrow=False,
                       font=dict(color=INK_SECONDARY, size=12))
fig_pr.update_layout(showlegend=False, **base_layout(300))
fig_pr.update_xaxes(title=dict(text="Rappel", font=dict(color=INK_MUTED)), range=[0, 1])
fig_pr.update_yaxes(title=dict(text="Precision", font=dict(color=INK_MUTED)), range=[0, 1.02])

# --- Q9 : cout metier vs seuil ---
fig_cout = go.Figure()
fig_cout.add_scatter(x=seuils_range, y=couts, mode="lines", line=dict(color=BLUE, width=2.5),
                      hovertemplate="Seuil %{x:.2f} - Cout %{y:.0f} EUR<extra></extra>")
fig_cout.add_vline(x=seuil_opt, line=dict(color=GOOD, width=2, dash="dash"))
fig_cout.add_vline(x=0.5, line=dict(color=INK_MUTED, width=1.5, dash="dash"))
fig_cout.add_annotation(x=seuil_opt, y=max(couts) * 0.96, text=f"Optimal {seuil_opt:.2f}",
                         showarrow=False, font=dict(color=GOOD, size=11), xanchor="left")
fig_cout.add_annotation(x=0.5, y=max(couts) * 0.8, text="Defaut 0.5",
                         showarrow=False, font=dict(color=INK_MUTED, size=11), xanchor="left")
fig_cout.update_layout(showlegend=False, **base_layout(300))
fig_cout.update_xaxes(title=dict(text="Seuil de decision", font=dict(color=INK_MUTED)))
fig_cout.update_yaxes(title=dict(text="Cout total (EUR)", font=dict(color=INK_MUTED)))

# --- Q10 : rappel par region ---
regions = list(recalls_region.keys())
region_values = list(recalls_region.values())
fig_region = go.Figure(go.Bar(
    x=regions, y=region_values, marker_color=ORANGE, width=0.4,
    text=[f"{v:.2f}" for v in region_values], textposition="outside",
    textfont=dict(color=INK_SECONDARY),
    hovertemplate="Region %{x} : rappel %{y:.3f}<extra></extra>",
))
fig_region.update_layout(**base_layout(300))
fig_region.update_yaxes(range=[0, 1.15])
fig_region.update_xaxes(title=dict(text="Region", font=dict(color=INK_MUTED)))

# --- MCC : meter divergent (HTML/CSS, pas de graphique Plotly) ---
if mcc >= 0.5:
    mcc_color, mcc_label = GOOD, "Bon pouvoir predictif"
elif mcc >= 0.2:
    mcc_color, mcc_label = WARNING, "Pouvoir predictif modere"
else:
    mcc_color, mcc_label = CRITICAL, "Faible pouvoir predictif"
mcc_fill_pct = abs(mcc) / 1.0 * 50
mcc_fill_side = "left: 50%;" if mcc >= 0 else f"right: 50%;"


def stat_tile(label, value, accent):
    return f"""
    <div class="tile" style="border-left-color:{accent}">
      <div class="tile-label">{label}</div>
      <div class="tile-value">{value}</div>
    </div>"""


def panel(title, body_html):
    return f"""
    <div class="panel">
      <h3>{title}</h3>
      {body_html}
    </div>"""


def insight_box(text_html):
    return f'<div class="insight">{text_html}</div>'


kpis = "".join([
    stat_tile("Transactions analysees", f"{len(df)}", INK_MUTED),
    stat_tile("Taux de fraude reel", f"{df.y_true.mean():.1%}", INK_MUTED),
    stat_tile("Accuracy (seuil 0.5)", f"{accuracy:.2f}", BLUE),
    stat_tile("Rappel (seuil 0.5)", f"{recall:.2f}", ORANGE),
    stat_tile("AUC-ROC", f"{auc_roc:.3f}", BLUE),
    stat_tile("Coefficient MCC", f"{mcc:.3f}", mcc_color),
])

economie = cout_defaut - cout_opt

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detection de fraude bancaire - tableau de bord</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: {PAGE};
    color: {INK};
    font-family: {FONT};
  }}
  .wrap {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 32px 24px 60px;
  }}
  header h1 {{
    font-size: 26px;
    margin: 0 0 6px;
  }}
  header p {{
    margin: 0 0 28px;
    color: {INK_SECONDARY};
    font-size: 14px;
  }}
  .kpis {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 14px;
    margin-bottom: 36px;
  }}
  .tile {{
    background: {SURFACE};
    border: 1px solid rgba(11,11,11,0.08);
    border-left-width: 4px;
    border-left-style: solid;
    border-radius: 10px;
    padding: 14px 16px;
  }}
  .tile-label {{
    font-size: 12px;
    color: {INK_SECONDARY};
    margin-bottom: 6px;
  }}
  .tile-value {{
    font-size: 24px;
    font-weight: 600;
  }}
  section {{ margin-bottom: 40px; }}
  section > h2 {{
    font-size: 18px;
    border-bottom: 1px solid rgba(11,11,11,0.10);
    padding-bottom: 8px;
    margin-bottom: 18px;
  }}
  .grid3 {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
  }}
  @media (max-width: 980px) {{
    .grid3 {{ grid-template-columns: 1fr; }}
  }}
  .panel {{
    background: {SURFACE};
    border: 1px solid rgba(11,11,11,0.08);
    border-radius: 12px;
    padding: 16px 16px 8px;
  }}
  .panel h3 {{
    font-size: 13px;
    font-weight: 600;
    color: {INK_SECONDARY};
    margin: 0 0 8px;
    text-transform: uppercase;
    letter-spacing: 0.02em;
  }}
  .insight {{
    grid-column: 1 / -1;
    background: {SURFACE};
    border: 1px solid rgba(11,11,11,0.08);
    border-left: 4px solid {INK_MUTED};
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13.5px;
    line-height: 1.55;
    color: {INK_SECONDARY};
  }}
  .insight b {{ color: {INK}; }}
  .meter-card {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 300px;
    padding: 0 8px;
  }}
  .meter-value {{
    font-size: 34px;
    font-weight: 600;
    margin-bottom: 4px;
  }}
  .meter-desc {{
    font-size: 13px;
    color: {INK_SECONDARY};
    margin-bottom: 18px;
  }}
  .meter-track {{
    position: relative;
    height: 10px;
    background: {GRID};
    border-radius: 6px;
  }}
  .meter-fill {{
    position: absolute;
    top: 0; bottom: 0;
    {mcc_fill_side}
    width: {mcc_fill_pct:.1f}%;
    background: {mcc_color};
    border-radius: 6px;
  }}
  .meter-mid {{
    position: absolute;
    left: 50%;
    top: -4px;
    width: 2px;
    height: 18px;
    background: {INK_MUTED};
  }}
  .meter-ticks {{
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: {INK_MUTED};
    margin-top: 6px;
  }}
  footer {{
    text-align: center;
    font-size: 12px;
    color: {INK_MUTED};
    margin-top: 20px;
  }}
</style>
</head>
<body>
<script>{pyo.get_plotlyjs()}</script>
<div class="wrap">
  <header>
    <h1>Detection de fraude bancaire - tableau de bord</h1>
    <p>{len(df)} transactions - taux de fraude reel {df.y_true.mean():.1%} - seuil de decision par defaut {SEUIL}</p>
  </header>

  <div class="kpis">{kpis}</div>

  <section>
    <h2>Partie 1 - Fondamentaux</h2>
    <div class="grid3">
      {panel("Matrice de confusion (seuil 0.5)", to_div(fig_cm))}
      {panel("Metriques au seuil 0.5", to_div(fig_metrics))}
      {panel("Accuracy trompeuse sur classes desequilibrees", to_div(fig_trap))}
      {insight_box(
          f"Le modele naif (toujours &laquo;&nbsp;non-fraude&nbsp;&raquo;) atteint <b>{accuracy_naif:.0%}</b> "
          f"d'accuracy sans detecter une seule fraude (rappel <b>{recall_naif:.0%}</b>), alors que le modele "
          f"teste detecte <b>{recall:.0%}</b> des fraudes. Sur des classes desequilibrees, l'accuracy seule ne "
          f"suffit pas a juger un modele : precision, rappel et F1 restent sensibles a la classe minoritaire."
      )}
    </div>
  </section>

  <section>
    <h2>Partie 2 - Seuils et courbes</h2>
    <div class="grid3">
      {panel("Precision / Rappel selon le seuil", to_div(fig_seuil))}
      {panel("Courbe ROC", to_div(fig_roc))}
      {panel("Courbe Precision-Rappel", to_div(fig_pr))}
      {insight_box(
          f"Baisser le seuil favorise le rappel au detriment de la precision, et inversement : le choix depend "
          f"du cout metier de chaque erreur, pas d'une convention arbitraire a 0.5. AUC-ROC (<b>{auc_roc:.3f}</b>) "
          f"parait excellente car elle integre les vrais negatifs, tres nombreux ici ; la PR-AUC (<b>{auc_pr:.3f}</b>) "
          f"se concentre sur la classe minoritaire et reflete mieux la difficulte reelle du probleme."
      )}
    </div>
  </section>

  <section>
    <h2>Partie 3 - Pour aller plus loin</h2>
    <div class="grid3">
      {panel("Coefficient MCC", f'''
        <div class="meter-card">
          <div class="meter-value" style="color:{mcc_color}">{mcc:.3f}</div>
          <div class="meter-desc">{mcc_label}</div>
          <div class="meter-track">
            <div class="meter-mid"></div>
            <div class="meter-fill"></div>
          </div>
          <div class="meter-ticks"><span>-1</span><span>0</span><span>+1</span></div>
        </div>
      ''')}
      {panel(f"Cout metier (FP={COUT_FP} EUR, FN={COUT_FN} EUR)", to_div(fig_cout))}
      {panel("Rappel par region (seuil 0.5)", to_div(fig_region))}
      {insight_box(
          f"Le MCC de <b>{mcc:.3f}</b> confirme un vrai pouvoir predictif, pas un simple effet de classe "
          f"majoritaire. Au seuil 0.5, le cout total est de <b>{cout_defaut:,.0f} EUR</b> ; le seuil "
          f"<b>{seuil_opt:.2f}</b> le ramene a <b>{cout_opt:,.0f} EUR</b> (economie de <b>{economie:,.0f} EUR</b>), "
          f"car une fraude ratee coute bien plus cher qu'une fausse alerte. Cote equite, le rappel passe de "
          f"<b>{recalls_region.get('A', 0):.0%}</b> en region A a <b>{recalls_region.get('B', 0):.0%}</b> en "
          f"region B : un ecart de cette ampleur est une discrimination silencieuse invisible dans les metriques "
          f"globales, a verifier par sous-groupe avant tout deploiement."
      ).replace(",", " ")}
    </div>
  </section>

  <footer>Dashboard realise par Riantsoa CHERICA</footer>
</div>
</body>
</html>
"""

OUTPUT_HTML.write_text(html, encoding="utf-8")
print(f"Tableau de bord genere : {OUTPUT_HTML}")

try:
    webbrowser.open(OUTPUT_HTML.resolve().as_uri())
except Exception:
    pass
