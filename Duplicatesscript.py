import os
import pandas as pd

# --- DUPLICATAS ---

# Configurações de caminho
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Carregar o DataFrame
df = pd.read_csv(os.path.join(BASE_PATH, "df.csv"))

print("--- Verificação de Duplicatas ---")

# 1. Duplicatas no arquivo de informações (linhas inteiras repetidas)
duplicatas_csv = df.duplicated().sum()
print(f"Linhas totalmente duplicadas no CSV: {duplicatas_csv}")

if duplicatas_csv > 0:
    print("Exemplos de linhas duplicadas:")
    print(df[df.duplicated()].head())

# 2. Duplicatas de caminhos de arquivos (mesma imagem em múltiplas linhas)
print("\nVerificando se arquivos se repetem em diferentes registros:")
for col in ["images", "masks", "collages"]:
    dups = df[col].duplicated().sum()
    print(f"Arquivos duplicados na coluna '{col}': {dups}")

# 3. Verificação cruzada (mesmo arquivo usado em colunas diferentes - erro comum)
all_paths = pd.concat([df['images'], df['masks'], df['collages']])
total_dups_geral = all_paths.duplicated().sum()
print(f"\nTotal de caminhos duplicados considerando todas as colunas: {total_dups_geral}")
