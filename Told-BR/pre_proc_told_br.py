# PROBELMA DE ENCODING/EMOJI/COREANO ... Tirar?

import re #regex
import nltk #pln 
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords

# download das stopwords
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words_pt = set(nltk_stopwords.words("portuguese")) # set ao inves de lista com as stopwords

###############################################################################

# função de pre-processamento textual
def pre_processar_told(texto: str, remover_stopwords: bool = False) -> str: # retirar ou não?? (ele tira o não?)

    if not isinstance(texto, str): # garante que dado é str
        return ""

    # lowercase -> minúsculas
    texto = texto.lower()

    # eemoção de URLs, e-mails, menções e hashtags -> re.sub(padrão, substituto, texto)
    texto = re.sub(r"http\S+|www\.\S+", "", texto) # URL's (que começam com http, https ou www)
    texto = re.sub(r"\S+@\S+\.\S+", "", texto) # emails
    texto = re.sub(r"@[A-Za-z0-9_.-]+", "", texto)  # menções (@ inclui pontos e hífens)
    texto = re.sub(r"#[A-Za-z0-9_.-]+", "", texto)  # Hashtags
    texto = re.sub(r"[^\w\s]", " ", texto) # pontuação, caracteres especiais e emojis
    texto = re.sub(r"_", " ", texto) # remove o caractere de underline explicitamente
    texto = re.sub(r"\s+", " ", texto).strip() # espaços excedentes

    # stopwords (parâmetro como True)
    if remover_stopwords:
        palavras = texto.split()
        texto = " ".join([p for p in palavras if p not in stop_words_pt])

    return texto

###############################################################################

# carregamento e limpeza
base_told = pd.read_csv("Told-BR/told_br.csv")

# binariza o rótulo: se a coluna 'misogyny' tiver valor >= 1.0), vira 1; senão, 0
base_told["misogyny_binary"] = (base_told["misogyny"] >= 1.0).astype(int)


# PRÉ-PROCESSAMENTO TEXTUAL

# sobrescreve a coluna 'text' com o texto limpo, aplica função a cada linha
base_told["text"] = base_told["text"].apply(
    lambda x: pre_processar_told(x, remover_stopwords=False)
    )

# remove registros que ficaram vazios após a limpeza
told_clean = base_told[base_told["text"] != ""].copy()

# agrupa os textos iguais e pega a maior nota de misoginia atribuída a ele e o maior valor binário
told_clean = told_clean.groupby("text", as_index=False).agg({"misogyny": "max", "misogyny_binary": "max"})

# deixar só essas colunas: text, misogyny e misogyny_binary
colunas_desejadas = ["text", "misogyny", "misogyny_binary"]
told_clean = told_clean[colunas_desejadas].copy()

# salvar dados tratados dentro da pasta Told-BR
told_clean.to_csv("Told-BR/told_br_clean.csv", index=False)

# prints
print("ToLD-Br Processado com Sucesso!!")
print(f"Total de Amostras Limpas: {len(told_clean)}")
print(f"Distribuição da Classe Misoginia:")
print(told_clean["misogyny_binary"].value_counts())
