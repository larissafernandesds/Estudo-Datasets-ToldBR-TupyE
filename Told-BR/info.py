# TOLD-BR: textos duplicados, desbalanceamento, conflito de rótulo, e escolher forma de binarizar!!

import pandas as pd

# carrega a base do ToLD-Br. Um tabelão...
base_told = pd.read_csv("Told-BR/told_br.csv")

print("=" * 50)
print("                     ToLD-Br")
print("=" * 50)

print(f"\nTotal de amostras na base: {len(base_told)}") # n° de linhas
print(f"\nCOLUNAS DISPONÍVEIS")
print(base_told.columns.tolist()) # lista com os nomes das colunas


print(f"\nDISTRIBUIÇÃO DA CLASSE BRUTA ({'misogyny'.upper()})") # letra maiúscula
print(base_told["misogyny"].value_counts()) # quantas vezes cada valor aparece
print("\nPERCENTUAL BRUTO") 
print(base_told["misogyny"].value_counts(normalize=True) * 100) # proporção e em seguida porcentagem


# Integridade dos dados
print("\nINTEGRIDADE DOS DADOS")
print(f"Valores ausentes em 'text': {base_told['text'].isnull().sum()}")
print(f"Textos duplicados no dataset: {base_told.duplicated(subset=['text']).sum()}") # verifica se o texto já apareceu anteriormente. A primeira ocorrência de cada texto recebe False. As ocorrências seguintes recebem True.


# 1 ou + anotadores (Misógino) | 0  (Não Misógino)
misoginia_um_ou_mais = base_told["misogyny"] >= 1.0 # Falsos e Verdadeiros

contagem = misoginia_um_ou_mais.value_counts()
porcentagem = misoginia_um_ou_mais.value_counts(normalize=True) * 100

print("\n" + "=" * 50)
print("       REGRA: 1 ou + anotadores (1) | 0 (0)")
print("=" * 50)
print("Quantidade de exemplos:")
print(contagem)
print("\nPorcentagem:")
print(f"Não Misógino (0): {porcentagem[False]:.2f}%")
print(f"Misógino (1, 2 ou 3): {porcentagem[True]:.2f}%")