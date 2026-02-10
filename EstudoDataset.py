import os
import pandas as pd

#INTEGRIDADE DOS ARQUIVOS

#verificar se os arquivos listados no DataFrame existem no diretório de dados
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")

df = pd.read_csv(os.path.join(BASE_PATH, "df.csv"))

missing_files = []

for col in ["images", "masks", "collages"]:
    for path in df[col]:
        full_path = os.path.join(DATA_PATH, path)
        if not os.path.isfile(full_path):
            missing_files.append(full_path)

print(f"Arquivos ausentes: {len(missing_files)}")
print(missing_files[:10])

#Todas as imagens, máscaras e colagens listadas no arquivo de informações existem no diretório correto.


#verificar formatos dos arquivos
from collections import Counter

def contar_formatos(coluna):
    extensoes = [os.path.splitext(p)[1].lower() for p in df[coluna]]
    return Counter(extensoes)

print("Formatos em images:", contar_formatos("images"))
print("Formatos em masks:", contar_formatos("masks"))
print("Formatos em collages:", contar_formatos("collages"))

#100% padronizado dentro de cada categoria, com imagens e máscaras em .png e colagens em .jpg.


