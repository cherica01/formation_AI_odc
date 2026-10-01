import sys
import rag
from rag import ask_fr, ask_detail

rag.DEBUG = False

TESTS_FR = [
    ("Combien de jours de congés payés par an ?", ["25"]),
    ("Combien de jours de télétravail par semaine ?", ["3", "trois"]),
    ("Quel est le jour de présence obligatoire ?", ["mardi"]),
    ("Quelle est l'indemnité de télétravail ?", ["20"]),
    ("Tous les combien de jours change-t-on son mot de passe ?", ["90"]),
    ("Quel est le numéro du support ?", ["4200"]),
    ("Quelle est la capitale de l'Australie ?", ["ne sais pas"]),
]

TESTS_MG = [
    ("Firy andro ny fialan-tsasatra isan-taona?", ["25"]),
    ("Firy andro isan'herinandro no azo atao ny asa avy any an-trano?", ["3", "telo"]),
    ("Andro inona no tsy maintsy tonga any am-piasana?", ["talata"]),
    ("Ohatrinona ny vola omena isam-bolana ho an'ny fitaovana amin'ny asa an-trano?", ["20"]),
    ("Isaky ny firy andro no tsy maintsy ovaina ny tenimiafina?", ["90"]),
    ("Inona ny laharana antsoina hifandraisana amin'ny fanohanana?", ["4200"]),
    ("Iza no lohan'ny fanjakana any Aostralia?", ["tsy fantatro"]),
]

def test_fr(q):
    rep, _ = ask_fr(q)
    return {"rep": rep}

def test_mg(q):
    rep, _, q_fr, r_fr = ask_detail(q)
    return {"rep": rep, "q_fr": q_fr, "r_fr": r_fr}

def evaluer(nom, tests, fn, sortie):
    def ecrire(texte=""):
        print(texte)
        sortie.write(texte + "\n")

    ecrire(f"\n===== {nom} =====")
    ok = 0
    for i, (q, attendus) in enumerate(tests, 1):
        r = fn(q)
        good = any(a.lower() in r["rep"].lower() for a in attendus)
        ok += good
        ecrire(f"\n{i}. [{'OK' if good else 'ECHEC'}] {q}")
        ecrire(f"     Réponse            : {r['rep']}")
        if "q_fr" in r:
            ecrire(f"     Question traduite  : {r['q_fr']}")
            ecrire(f"     Réponse en français: {r['r_fr']}")
    ecrire(f"\n>>> Score {nom} : {ok}/{len(tests)}")

mode = sys.argv[1] if len(sys.argv) > 1 else "both"

with open("rapport_eval.txt", "w", encoding="utf-8") as sortie:
    if mode in ("fr", "both"):
        evaluer("FRANCAIS", TESTS_FR, test_fr, sortie)
    if mode in ("mg", "both"):
        evaluer("MALGACHE", TESTS_MG, test_mg, sortie)

print("\nRapport enregistré dans rapport_eval.txt")