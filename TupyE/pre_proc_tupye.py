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


def pre_processar_tupye(texto: str, remover_stopwords: bool = False) -> str:
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
train_df = pd.read_csv("TupyE/multilabel_train.csv")
test_df = pd.read_csv("TupyE/multilabel_test.csv")

# remover duplicatas internas
train_clean = train_df.drop_duplicates(subset=["text"], keep="first").copy()
test_clean = test_df.drop_duplicates(subset=["text"], keep="first").copy()

# eliminar Data Leakage (remover do teste o que já existe no treino)
test_clean = test_clean[~test_clean["text"].isin(train_clean["text"])].copy()

# PRÉ-PROCESSAMENTO TEXTUAL
train_clean["text_clean"] = train_clean["text"].apply(
    lambda x: pre_processar_tupye(x, remover_stopwords=False)
)
test_clean["text_clean"] = test_clean["text"].apply(
    lambda x: pre_processar_tupye(x, remover_stopwords=False)
)

# remover registros que ficaram vazios após a limpeza
train_clean = train_clean[train_clean["text_clean"] != ""].copy()
test_clean = test_clean[test_clean["text_clean"] != ""].copy()

# SALVAR DADOS TRATADOS
train_clean.to_csv("tupye_train_clean.csv", index=False)
test_clean.to_csv("tupye_test_clean.csv", index=False)

print("=== TuPyE Processado com Sucesso ===")
print(f"Treino Final: {len(train_clean)} amostras")
print(f"Teste Final:  {len(test_clean)} amostras")
