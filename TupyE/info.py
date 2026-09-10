# TUPY-E: textos duplicados, 2 com inconsistencia de rótulo, desbalanceamento, já esta dividida em treino/teste!!

import pandas as pd

# carrega os arquivos do diretório 
train_df = pd.read_csv("TupyE/multilabel_train.csv")
test_df = pd.read_csv("TupyE/multilabel_test.csv")

print(f" Total de amostras de TREINO: {len(train_df)}")
print(f" Total de amostras de TESTE: {len(test_df)}")

print("\n--- TODAS AS COLUNAS DISPONÍVEIS ---")
print(train_df.columns.tolist())

print("\n--- PRIMEIRAS 2 LINHAS DO TREINO ---")
print(train_df.head(2))

print("\n--- MISOGYNY NO TREINO ---")
print(train_df["misogyny"].value_counts().sort_index())

print("\n--- PERCENTUAL DE CADA VALOR - TREINO  ---")
print(train_df["misogyny"].value_counts(normalize=True).sort_index() * 100)

print("\n--- MISOGYNY NO TESTE ---")
print(test_df["misogyny"].value_counts().sort_index())

print("\n--- PERCENTUAL DE CADA VALOR - TESTE ---")
print(test_df["misogyny"].value_counts(normalize=True).sort_index() * 100)

print("\n--- TIPOS DAS COLUNAS ---")
print(train_df.dtypes)

print("\n--- VALORES AUSENTES ---")
print(train_df.isnull().sum())

# duplicados dentro do treino
print("Duplicados no treino:",
      train_df["text"].duplicated().sum())

# duplicados dentro do teste
print("Duplicados no teste:",
      test_df["text"].duplicated().sum())

# textos iguais entre treino e teste
repetidos = set(train_df["text"]) & set(test_df["text"])

print("Repetidos entre treino e teste:",
      len(repetidos))

# total de amostras
total = len(train_df) + len(test_df)

# percentual de treino e teste
percentual_treino = len(train_df) / total * 100
percentual_teste = len(test_df) / total * 100

print(f"Total de amostras: {total}")
print(f"Treino: {len(train_df)} ({percentual_treino:.2f}%)")
print(f"Teste: {len(test_df)} ({percentual_teste:.2f}%)")

# textos que aparecem tanto no treino quanto no teste
repetidos = set(train_df["text"]) & set(test_df["text"])

print(f"Quantidade de textos repetidos entre treino e teste: {len(repetidos)}")

# mostrar alguns exemplos
for texto in list(repetidos)[:10]:
    print("\n---")
    print(texto)

    print("Treino:")
    print(train_df[train_df["text"] == texto][["misogyny", "researcher", "year", "source"]])

    print("Teste:")
    print(test_df[test_df["text"] == texto][["misogyny", "researcher", "year", "source"]])


# verificar os textos que aparecem nos dois conjuntos
# e possuem rótulos diferentes

conflitos = []

for texto in repetidos:
    rotulos_treino = set(
        train_df[train_df["text"] == texto]["misogyny"]
    )

    rotulos_teste = set(
        test_df[test_df["text"] == texto]["misogyny"]
    )

    if rotulos_treino != rotulos_teste:
        conflitos.append(texto)

print("Textos duplicados entre treino e teste:", len(repetidos))
print("Textos duplicados com conflito de rótulo:", len(conflitos))

print("\n--- EXEMPLOS DE CONFLITOS ---")

for texto in conflitos[:20]:
    print("\nTexto:", texto)

    print("Treino:")
    print(train_df[train_df["text"] == texto][
        ["misogyny", "researcher", "year", "source"]
    ])

    print("Teste:")
    print(test_df[test_df["text"] == texto][
        ["misogyny", "researcher", "year", "source"]
    ])