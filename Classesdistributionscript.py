import os
import pandas as pd

# --- DISTRIBUIÇÃO DAS CLASSES ---

# Configurações de caminho
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Carregar o DataFrame
df = pd.read_csv(os.path.join(BASE_PATH, "df.csv"))

print("--- Análise de Distribuição ---")

# Nota: Como o df.csv padrão não possui uma coluna 'label', 
# este código verifica se ela existe antes de tentar contar.
# Se o seu dataset usar outro nome (ex: 'categoria', 'class'), altere abaixo.

target_column = 'label' 

if target_column in df.columns:
    print(f"\nDistribuição da coluna '{target_column}':")
    contagem = df[target_column].value_counts()
    percentual = df[target_column].value_counts(normalize=True) * 100
    
    df_dist = pd.DataFrame({'Quantidade': contagem, 'Percentual (%)': percentual})
    print(df_dist)
else:
    print(f"\nA coluna '{target_column}' não foi encontrada no CSV.")
    print("Colunas disponíveis no arquivo:", list(df.columns))
    print("\nDica: Se houver classes, altere a variável 'target_column' para o nome correto da coluna.")

print(f"\nTotal de registros no dataset: {len(df)}")
