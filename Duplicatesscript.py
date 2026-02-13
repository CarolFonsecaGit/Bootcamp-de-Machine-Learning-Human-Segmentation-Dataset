import os
import pandas as pd

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_PATH, "df.csv"), index_col=0)

print("--- Verificação de Duplicatas ---")

print("\nColunas do CSV:")
print(df.columns)

# 1️⃣ Duplicatas de linhas completas
duplicatas_csv = df.duplicated().sum()
print(f"\nLinhas totalmente duplicadas no CSV: {duplicatas_csv}")

# 2️⃣ Duplicatas por coluna
print("\nDuplicatas por coluna:")
for col in ["images", "masks", "collages"]:
    dups = df[col].duplicated().sum()
    print(f"{col}: {dups}")

# 3️⃣ Verificação cruzada
all_paths = pd.concat([df['images'], df['masks'], df['collages']])
total_dups = all_paths.duplicated().sum()
print(f"\nDuplicatas considerando todas as colunas: {total_dups}")
