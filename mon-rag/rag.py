import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import os
os.environ["HF_HUB_OFFLINE"] = "1"                 # le modèle est déjà téléchargé
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DEBUG = True   # affiche les étapes intermédiaires ; mettre False à la fin

emb = OllamaEmbeddings(model="nomic-embed-text")
vs = Chroma(persist_directory="db", embedding_function=emb)
retriever = vs.as_retriever(search_kwargs={"k": 3})
llm = ChatOllama(model="llama3.2", temperature=0)

# ---------- RAG en français ----------
prompt = ChatPromptTemplate.from_template("""Tu es l'assistant RH de Zenith Tech.
Réponds uniquement à partir du contexte ci-dessous, en français, de façon concise,
avec des phrases courtes et simples.
Ne déduis rien : si l'information n'est pas écrite explicitement dans le contexte,
même pour répondre "non", réponds exactement : "Je ne sais pas d'après les documents."

Contexte :
{context}

Question : {question}
Réponse :""")

chain = prompt | llm | StrOutputParser()

def fmt(docs):
    return "\n\n".join(d.page_content for d in docs)

def ask_fr(question):
    """Question en français -> réponse en français."""
    docs = retriever.invoke(question)
    answer = chain.invoke({"context": fmt(docs), "question": question})
    sources = sorted({d.metadata.get("source", "?") for d in docs})
    return answer, sources

# ---------- Couche malgache (NLLB) ----------
NLLB = "facebook/nllb-200-distilled-600M"
_tok = None
_model = None

def _charger_nllb():
    """Charge NLLB une seule fois, au premier besoin."""
    global _tok, _model
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        _tok = AutoTokenizer.from_pretrained(NLLB)
        _model = AutoModelForSeq2SeqLM.from_pretrained(NLLB)
def traduire(texte, src, tgt):
    _charger_nllb()
    _tok.src_lang = src
    entree = _tok(texte, return_tensors="pt")
    sortie = _model.generate(
        **entree,
        forced_bos_token_id=_tok.convert_tokens_to_ids(tgt),
        max_new_tokens=200,
    )
    return _tok.batch_decode(sortie, skip_special_tokens=True)[0].strip()

def mg_to_fr(texte):
    return traduire(texte, "plt_Latn", "fra_Latn")

def fr_to_mg(texte):
    return traduire(texte, "fra_Latn", "plt_Latn")

REFUS_MG = "Tsy fantatro araka ny antontan-taratasy."   # à corriger par vous

def ask_detail(question_mg):
    """Question MG -> (réponse MG, sources, question FR, réponse FR)."""
    question_fr = mg_to_fr(question_mg)
    reponse_fr, sources = ask_fr(question_fr)
    if "ne sais pas" in reponse_fr.lower():
        reponse_mg = REFUS_MG
    else:
        reponse_mg = fr_to_mg(reponse_fr)
    return reponse_mg, sources, question_fr, reponse_fr

def ask(question_mg):
    """Question en malgache -> réponse en malgache."""
    reponse_mg, sources, question_fr, reponse_fr = ask_detail(question_mg)
    if DEBUG:
        print("  [question traduite ]", question_fr)
        print("  [réponse en français]", reponse_fr)
    return reponse_mg, sources