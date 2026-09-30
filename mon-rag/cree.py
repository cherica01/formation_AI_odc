import os
os.makedirs("data", exist_ok=True)

docs = {
"conges.txt": """Politique de congés de Zenith Tech.
Chaque salarié bénéficie de 25 jours de congés payés par an.
Les congés doivent être demandés au moins 15 jours à l'avance via le portail RH.
Le report des congés non pris est limité à 5 jours et doit être utilisé avant le 31 mars de l'année suivante.
Un congé de plus de 10 jours consécutifs nécessite l'accord du directeur de service.
Congés spéciaux : mariage 5 jours, naissance 3 jours, décès d'un proche 3 jours.
""",
"teletravail.txt": """Charte du télétravail de Zenith Tech.
Le télétravail est autorisé jusqu'à 3 jours par semaine après 6 mois d'ancienneté.
Le jour de présence obligatoire en équipe est le mardi.
Le salarié reçoit une indemnité de 20 euros par mois pour ses frais d'équipement.
Le matériel fourni comprend un ordinateur portable et un écran.
Le VPN est obligatoire pour accéder aux serveurs internes.
""",
"support.txt": """Support informatique de Zenith Tech.
Le support est joignable au poste 4200 ou par email à support@zenith.example, du lundi au vendredi de 8h à 18h.
Les mots de passe doivent être changés tous les 90 jours et contenir au moins 12 caractères.
""",
}

for nom, texte in docs.items():
    with open(os.path.join("data", nom), "w", encoding="utf-8") as f:
        f.write(texte)
print("Corpus créé dans data/")