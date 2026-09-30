from rag import ask

SCENARIO = [
    # --- Les 4 questions d'origine ---
    "Combien de jours de congés payés ai-je par an ?",
    "Quel jour dois-je venir au bureau ?",
    "Comment joindre le support ?",
    "Qui a gagné la Coupe du monde 2018 ?",

    # --- Questions précises (valeur chiffrée) ---
    "Combien d'euros par mois reçoit-on pour les frais d'équipement en télétravail ?",
    "Au bout de combien de jours de préavis faut-il demander ses congés ?",
    "Tous les combien de jours doit-on changer son mot de passe ?",

    # --- Questions reformulées (autres mots, même sens) ---
    "Combien de jours de vacances ai-je chaque année ?",
    "Combien de jours par semaine puis-je travailler depuis chez moi ?",
    "À quelle heure ferme le service d'assistance informatique ?",

    # --- Questions pièges (information absente) ---
    "Quel est le salaire moyen chez Zenith Tech ?",
    "Combien de jours de congé pour un déménagement ?",
    "Y a-t-il une cantine dans l'entreprise ?",

    # --- Questions composées (deux informations) ---
    "Combien de jours de congés ai-je et quel est le numéro du support ?",
    "Quel jour dois-je être présent et combien de jours de télétravail ai-je droit ?",
]

with open("transcript.txt", "w", encoding="utf-8") as f:
    for q in SCENARIO:
        answer, sources = ask(q)
        bloc = f"Vous > {q}\nBot  > {answer}\n       [ {', '.join(sources)} ]\n"
        print(bloc)          # affichage (peut rester déformé à l'écran)
        f.write(bloc + "\n") # fichier propre en UTF-8