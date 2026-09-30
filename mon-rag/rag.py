from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DEBUG = True   # affiche les étapes intermédiaires ; mettre False à la fin

emb = OllamaEmbeddings(model="nomic-embed-text")
vs = Chroma(persist_directory="db", embedding_function=emb)
retriever = vs.as_retriever(search_kwargs={"k": 3})
llm = ChatOllama(model="llama3.2", temperature=0)

# ---------- RAG en français (inchangé, sauf la consigne "ne déduis rien") ----------
prompt = ChatPromptTemplate.from_template("""Tu es l'assistant RH de Zenith Tech.
Réponds uniquement à partir du contexte ci-dessous, en français, de façon concise.
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
    """Ancienne fonction : question en français -> réponse en français."""
    docs = retriever.invoke(question)
    answer = chain.invoke({"context": fmt(docs), "question": question})
    sources = sorted({d.metadata.get("source", "?") for d in docs})
    return answer, sources

# ---------- Couche malgache ----------
REGLES = ("Réponds uniquement par la traduction, sans commentaire ni explication. "
          "Ne modifie jamais les nombres, les adresses email, les noms propres "
          "(Zenith Tech) ni les numéros de poste.")

mg_to_fr = (ChatPromptTemplate.from_template(
    "Traduis ce texte du malgache vers le français. " + REGLES +
    "\n\nTexte : {texte}\nTraduction :") | llm | StrOutputParser())

fr_to_mg = (ChatPromptTemplate.from_template(
    "Traduis ce texte du français vers le malgache. " + REGLES +
    "\n\nTexte : {texte}\nTraduction :") | llm | StrOutputParser())

REFUS_MG = "Tsy fantatro araka ny antontan-taratasy."   # à corriger par vous

def ask(question_mg):
    """Question en malgache -> réponse en malgache."""
    question_fr = mg_to_fr.invoke({"texte": question_mg}).strip()
    reponse_fr, sources = ask_fr(question_fr)

    if "ne sais pas" in reponse_fr.lower():
        reponse_mg = REFUS_MG
    else:
        reponse_mg = fr_to_mg.invoke({"texte": reponse_fr}).strip()


    return reponse_mg, sources