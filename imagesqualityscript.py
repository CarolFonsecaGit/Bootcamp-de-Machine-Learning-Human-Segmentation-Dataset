import os
import pandas as pd
from PIL import Image

# --- QUALIDADE DAS IMAGENS ---

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "segmentation_full_body_tik_tok_2615_img")

df = pd.read_csv(os.path.join(BASE_PATH, "df.csv"), index_col=0)

corrupted_images = []
total_checked = 0
column_counts = {}

print("Verificando integridade dos arquivos de imagem...\n")

for col in ["images", "masks", "collages"]:
    column_counts[col] = 0
    
    for path in df[col]:
        full_path = os.path.join(DATA_PATH, path)
        total_checked += 1
        column_counts[col] += 1

        if os.path.isfile(full_path):
            try:
                with Image.open(full_path) as img:
                    img.verify()
            except Exception as e:
                corrupted_images.append(f"{col} - {path}: {e}")
        else:
            corrupted_images.append(f"{col} - {path}: Arquivo não encontrado")

# --- RESULTADOS ---

print("📊 RESUMO DA ANÁLISE")
print("-" * 40)

for col, count in column_counts.items():
    print(f"{col}: {count} arquivos analisados")

print("-" * 40)
print(f"Total geral analisado: {total_checked}")
print(f"Total de arquivos com problema: {len(corrupted_images)}")

if corrupted_images:
    print("\nArquivos com problema (primeiros 10):")
    for img in corrupted_images[:10]:
        print(f"- {img}")
else:
    print("\n✅ Todas as imagens foram verificadas com sucesso.")
