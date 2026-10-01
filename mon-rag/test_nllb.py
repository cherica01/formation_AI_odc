from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

NOM = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(NOM)
model = AutoModelForSeq2SeqLM.from_pretrained(NOM)

def traduire(texte, src, tgt):
    tok.src_lang = src
    entree = tok(texte, return_tensors="pt")
    sortie = model.generate(
        **entree,
        forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
        max_new_tokens=200,
    )
    return tok.batch_decode(sortie, skip_special_tokens=True)[0]

print("--- MG -> FR ---")
for q in ["Firy andro ny fialan-tsasatra isan-taona?",
          "Andro inona no tsy maintsy tonga any am-piasana?",
          "Misy kantine ve ao amin'ny orinasa?"]:
    print(q, "->", traduire(q, "plt_Latn", "fra_Latn"))

print("\n--- FR -> MG ---")
for t in ["Vous avez 25 jours de congés payés par an.",
          "Le support est joignable au poste 4200.",
          "Le jour de présence obligatoire est le mardi."]:
    print(t, "->", traduire(t, "fra_Latn", "plt_Latn"))