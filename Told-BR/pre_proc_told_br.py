import re
import nltk
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords

# download das stopwords
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words_pt = set(nltk_stopwords.words("portuguese"))


def pre_processar_told(texto: str, remover_stopwords: bool = False) -> str:
    if not isinstance(texto, str):
        return ""

    # lowercase
    texto = texto.lower()

    # remoção de URLs, e-mails, menções e hashtags
    texto = re.sub(r"http\S+|www\.\S+", "", texto)
    texto = re.sub(r"\S+@\S+\.\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"#\w+", "", texto)

    # pontuação e especiais (MANTÉM NÚMEROS: \d e \w)
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"_", " ", texto)

    # espaços excedentes
    texto = re.sub(r"\s+", " ", texto).strip()

    # stopwords opcional
    if remover_stopwords:
        palavras = texto.split()
        texto = " ".join([p for p in palavras if p not in stop_words_pt])

    return texto


# CARREGAMENTO E LIMPEZA DE INTEGRIDADE

# 
told_df = pd.read_csv("Told-BR/told_br.csv")

# remover duplicatas globais de texto
told_clean = told_df.drop_duplicates(subset=["text"], keep="first").copy()

# binarização da classe de misoginia (Regra: >= 1 anotador vira classe 1)
told_clean["misogyny_binary"] = (told_clean["misogyny"] >= 1.0).astype(int)

# PRÉ-PROCESSAMENTO TEXTUAL 
told_clean["text_clean"] = told_clean["text"].apply(
    lambda x: pre_processar_told(x, remover_stopwords=False)
)

# remover registros que ficaram vazios após a limpeza
told_clean = told_clean[told_clean["text_clean"] != ""].copy()

# SALVAR DADOS TRATADOS
told_clean.to_csv("told_br_clean.csv", index=False)

print("=== ToLD-Br Processado com Sucesso ===")
print(f"Total de Amostras Limpas: {len(told_clean)}")
print(f"Distribuição da Classe Misoginia:")
print(told_clean["misogyny_binary"].value_counts())