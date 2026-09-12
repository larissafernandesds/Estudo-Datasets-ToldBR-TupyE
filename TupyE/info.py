# TUPY-E: textos duplicados, 2 com inconsistencia de rótulo, desbalanceamento, já esta dividida em treino/teste!!

import pandas as pd

# carrega os arquivos do diretório
treino_tupye = pd.read_csv("TupyE/multilabel_train.csv")
teste_tupye = pd.read_csv("TupyE/multilabel_test.csv")

print("=" * 50)
print("                     TuPyE")
print("=" * 50)

# n° de linhas
print(f"\nTotal de amostras de TREINO: {len(treino_tupye)}")
print(f"Total de amostras de TESTE: {len(teste_tupye)}")


# lista com os nomes das colunas
print("\nCOLUNAS DISPONÍVEIS")
print(treino_tupye.columns.tolist())


# distribuição da classe no treino
# proporção e em seguida porcentagem
print("\nDISTRIBUIÇÃO DA CLASSE (MISOGYNY) - TREINO")
print(f"{treino_tupye['misogyny'].value_counts()} {treino_tupye['misogyny'].value_counts(normalize=True).sort_index() * 100}") #quantas vezes cada valor aparece


# distribuição da classe no teste
# proporção e em seguida porcentagem
print("\nDISTRIBUIÇÃO DA CLASSE (MISOGYNY) - TESTE")
print(f"{teste_tupye['misogyny'].value_counts()} {teste_tupye['misogyny'].value_counts(normalize=True).sort_index() * 100}") #quantas vezes cada valor aparece


# integridade dos dados
print("\nINTEGRIDADE DOS DADOS")
print(f"Valores ausentes em 'text' no treino: {treino_tupye['text'].isnull().sum()}")
print(f"Valores ausentes em 'text' no teste: {teste_tupye['text'].isnull().sum()}")

# verifica se o texto já apareceu anteriormente. A primeira ocorrência de cada texto recebe False. As ocorrências seguintes recebem True.
print(f"Textos duplicados no treino: {treino_tupye.duplicated(subset=['text']).sum()}")
print(f"Textos duplicados no teste: {teste_tupye.duplicated(subset=['text']).sum()}")


# textos que aparecem tanto no treino quanto no teste
repetidos = set(treino_tupye["text"]) & set(teste_tupye["text"])
print(f"Textos repetidos entre treino e teste: {len(repetidos)}")


# total de amostras e proporção treino/teste
total = len(treino_tupye) + len(teste_tupye)
percentual_treino = len(treino_tupye) / total * 100
percentual_teste = len(teste_tupye) / total * 100

print("\n" + "=" * 50)
print("              DIVISÃO DA BASE")
print("=" * 50)

print(f"Total de amostras: {total}")
print(f"Treino: {len(treino_tupye)} ({percentual_treino:.2f}%)")
print(f"Teste: {len(teste_tupye)} ({percentual_teste:.2f}%)")