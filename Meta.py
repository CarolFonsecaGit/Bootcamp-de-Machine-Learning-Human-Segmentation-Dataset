import os
import pandas as pd
from PIL import Image

# --- CONSISTÊNCIA DOS METADADOS ---

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "segmentation_full_body_tik_tok_2615_img")

df = pd.read_csv(os.path.join(BASE_PATH, "df.csv"), index_col=0)

print("--- Verificação de Valores Ausentes ---")
print(df.isnull().sum())

print("\n--- Verificação de Inconsistência de Dimensões ---")

inconsistent_dimensions = []
files_checked = 0

for index, row in df.iterrows():

    image_path = os.path.join(DATA_PATH, row["images"])
    mask_path = os.path.join(DATA_PATH, row["masks"])
    collage_path = os.path.join(DATA_PATH, row["collages"])

    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        with Image.open(mask_path) as mask:
            mask_width, mask_height = mask.size

        with Image.open(collage_path) as collage:
            col_width, col_height = collage.size

        files_checked += 3

        # Verifica se máscara tem mesma dimensão da imagem
        if (mask_width, mask_height) != (img_width, img_height):
            inconsistent_dimensions.append({
                "arquivo": row["masks"],
                "esperado": (img_width, img_height),
                "encontrado": (mask_width, mask_height)
            })

        # Verifica se colagem tem mesma dimensão da imagem
        if (col_width, col_height) != (img_width, img_height):
            inconsistent_dimensions.append({
                "arquivo": row["collages"],
                "esperado": (img_width, img_height),
                "encontrado": (col_width, col_height)
            })

    except Exception:
        continue

print(f"\nTotal de arquivos verificados: {files_checked}")
print(f"Total de inconsistências de dimensão: {len(inconsistent_dimensions)}")

if inconsistent_dimensions:
    print("\nExemplos de inconsistências:")
    for item in inconsistent_dimensions[:5]:
        print(f"Arquivo: {item['arquivo']} | Esperado: {item['esperado']} | Encontrado: {item['encontrado']}")
else:
    print("\n✅ Todas as imagens, máscaras e colagens possuem dimensões consistentes.")
