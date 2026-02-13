import os
import pandas as pd
from PIL import Image

# --- CONSISTÊNCIA DOS METADADOS ---

# Configurações de caminho
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")

# Carregar o DataFrame
df = pd.read_csv(os.path.join(BASE_PATH, "df.csv"))

print("--- Verificação de Valores Ausentes ---")
print(df.isnull().sum())

print("\n--- Verificação de Inconsistência de Dimensões ---")
inconsistent_dimensions = []
image_dimensions_map = {}

for index, row in df.iterrows():
    for col in ["images", "masks", "collages"]:
        full_path = os.path.join(DATA_PATH, row[col])
        
        if os.path.isfile(full_path):
            try:
                with Image.open(full_path) as img:
                    width, height = img.size
                    
                    if col == "images":
                        image_dimensions_map[index] = (width, height)
                    elif index in image_dimensions_map:
                        if (width, height) != image_dimensions_map[index]:
                            inconsistent_dimensions.append({
                                "arquivo": row[col],
                                "esperado": image_dimensions_map[index],
                                "encontrado": (width, height)
                            })
            except:
                continue

print(f"Total de inconsistências de dimensão (máscara/colagem vs imagem): {len(inconsistent_dimensions)}")
if inconsistent_dimensions:
    print("Exemplos de inconsistências:")
    for item in inconsistent_dimensions[:5]:
        print(f"Arquivo: {item['arquivo']} | Esperado: {item['esperado']} | Encontrado: {item['encontrado']}")
